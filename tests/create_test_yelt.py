"""Generate a YELT for the tests"""
import os
import pandas as pd
import numpy as np

THIS_DIR = os.path.dirname(__file__)
print(THIS_DIR)


def poisson_pareto_yelt(random_state, freq0, n_years, min_loss, alpha):
    """Simulate and create a Poisson/Pareto YELT"""
    # Calculate number of events
    n_events = random_state.poisson(freq0 * n_years, 1)[0]
    print(f"n = {n_events} events")

    # Calculate time at which events occur
    event_times = random_state.uniform(low=1.0, high=n_years + 1, size=n_events)
    print(f"event time range: {event_times.min():.2f} -- " +
          f"{event_times.max():.2f}")

    # Calculate loss sizes
    losses = min_loss * (1 + random_state.pareto(alpha, n_events))
    print(f"Loss from {losses.min():.2g} -- {losses.max():.2g}")

    # Build the DataFrame
    yelt = pd.DataFrame({
        'Year': event_times.astype(int),
        'EventID': range(1, n_events + 1),
        'DayOfYear': 1 + np.int64((365 * np.mod(event_times, 1.0))),
        'Loss': losses,
    })

    # Sort based on time
    yelt = yelt.sort_values(['Year', 'DayOfYear', 'EventID', 'Loss'])

    # Convert to a series
    yelt = yelt.set_index(['Year', 'EventID', 'DayOfYear'],
                          verify_integrity=True)

    return yelt


def write_file(yelt, ofilename, **kwargs):
    """Write a series or dataframe to file and print message"""
    yelt.to_csv(ofilename, **kwargs)
    print(f"Written to {ofilename}")


def combine_yelts_two_models(yelt1, yelt2):
    """Combine YELTs from two models into one"""
    # Stack the YELTs together
    yelt_combined = pd.concat([yelt1, yelt2], axis=0, keys=['Model1', 'Model2'],
                              names=['ModelID'])

    # Add a net loss after a limit
    yelt_combined = yelt_combined['Loss'].rename('GrossLoss').to_frame()

    return yelt_combined


def split_gross_net(yelt, attach, limit):
    """Create a net and a gross loss by applying layer attachment and limit"""
    yelt = yelt.rename(columns={'Loss': 'GrossLoss'})

    yelt['NetLoss'] = yelt['GrossLoss']

    # Apply deductible
    yelt['NetLoss'] = (yelt['NetLoss'] - attach).clip(lower=0.0)

    # Apply a limit
    yelt['NetLoss'] = yelt['NetLoss'].clip(upper=limit)

    return yelt


def create_region_splits(yealt, rng):
    """Split out existing loss in a yelt randomly into different regions"""

    # Random split between three different regions
    runif = rng.random(size=(len(yealt), 2))
    runif.sort(axis=1)
    region_splits = np.hstack([runif[:, [0]], np.diff(runif, axis=1), 1 - runif[:, [1]]])

    # Apply percentage split to each loss
    for i in range(region_splits.shape[1]):
        yealt[i + 1] = yealt['GrossLoss'] * region_splits[:, i]

    # Collapse into single loss row
    yealt.columns.rename('RegionID', inplace=True)
    yealt = yealt.drop('GrossLoss', axis=1).stack().rename('Loss').to_frame()
    print(f"Initially {len(yealt):,.0f} rows for regions")
    print(f"Total Loss: {yealt['Loss'].sum():,.0f}")

    # Replace some region IDs, so we don't have the same 3 regions for each
    # loss.
    n_replace = 10000
    idx_replace = rng.choice(range(len(yealt)), replace=False, size=n_replace)
    new_regionids = yealt.index.get_level_values('RegionID').values
    new_regionids[idx_replace] = 4

    index_names = yealt.index.names
    yealt = yealt.reset_index()

    yealt['RegionID'] = new_regionids
    print(f"Total Loss: {yealt['Loss'].sum():,.0f}")

    return yealt.set_index(index_names)


def main():
    """Main script"""

    seed = 12345
    n_years = 1e5  # Number of years
    min_loss = 1e3

    rng = np.random.default_rng(seed=seed)

    yelt = poisson_pareto_yelt(rng,
                               freq0=0.5,  # Number of events per year
                               n_years=n_years,
                               min_loss=min_loss,
                               alpha=1.0)

    # Write to file
    write_file(yelt, "_data/example_pareto_poisson_yelt.csv", index=True)

    # Create a second YELT with two models and two losses
    yelt2 = poisson_pareto_yelt(rng,
                                freq0=1.0,  # Number of events per year
                                n_years=n_years,
                                min_loss=min_loss,
                                alpha=1.5)

    yelt2 = combine_yelts_two_models(yelt, yelt2)

    # Create Gross and Net columns by applying an attachment and limit
    yelt2 = split_gross_net(yelt2, 20e3, 120e3)

    # Write to file
    write_file(yelt2, "_data/example_two_models_grossnet.csv", index=True)

    # Create a Year Event Allocated Loss Table where event loss has been split
    print("Creating an allocated YELT among regions and loss sources")
    yealt = yelt2[['GrossLoss']].copy()
    print(f"Total Loss: {yealt['GrossLoss'].sum():,.0f}")

    yealt = create_region_splits(yealt, rng)

    # Add a source of loss ID randomly to each loss
    yealt['LossSourceID'] = rng.integers(low=1, high=6, size=len(yealt))
    yealt = yealt.set_index('LossSourceID', append=True)

    # Get rid of any duplicates
    yealt = yealt.groupby(yealt.index.names).sum()
    print(yealt.groupby(['RegionID', 'LossSourceID']).sum().unstack('RegionID'))
    print(f"Finally {len(yealt):,.0f} rows for regions and loss sources")

    write_file(yealt, "_data/example_allocated_loss.csv", index=True)


if __name__ == "__main__":
    main()
