"""Generate a YELT for the tests"""
import os
import pandas as pd
import numpy as np

THIS_DIR = os.path.dirname(__file__)
print(THIS_DIR)

def poisson_pareto_yelt(rs, freq0, n_years, min_loss, alpha):
    """Simulate and create a Poisson/Pareto YELT"""
    # Calculate number of events
    n_events = rs.poisson(freq0 * n_years, 1)[0]
    print(f"n = {n_events} events")

    # Calculate time at which events occur
    event_times = rs.uniform(low=1.0, high=n_years + 1, size=n_events)
    print(f"event time range: {event_times.min():.2f} -- " +
          f"{event_times.max():.2f}")

    # Calculate loss sizes
    losses = min_loss * (1 + rs.pareto(alpha, n_events))
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


def main():
    """Main script"""

    seed = 12345
    n_years = 1e5  # Number of years
    min_loss = 1e3
    ofilename = "_data/example_pareto_poisson_yelt.csv"

    rs = np.random.RandomState(seed=seed)

    yelt = poisson_pareto_yelt(rs,
                               freq0=0.5,  # Number of events per year
                               n_years=n_years,
                               min_loss=min_loss,
                               alpha=1.0)

    # Write to file
    yelt.to_csv(ofilename, index=True)
    print(f"Written to {ofilename}")

    # Create a second YELT with two models and two losses
    ofilename2 = "_data/example_two_models_grossnet.csv"
    yelt2 = poisson_pareto_yelt(rs,
                               freq0=1.0,  # Number of events per year
                               n_years=n_years,
                               min_loss=min_loss,
                               alpha=1.5)

    # Stack the YELTs together
    yelt_combined = pd.concat([yelt, yelt2], axis=0, keys=['Model1', 'Model2'],
                              names=['ModelID'])

    # Add a net loss after a limit
    yelt_combined = yelt_combined['Loss'].rename('GrossLoss').to_frame()
    yelt_combined['NetLoss'] = yelt_combined['GrossLoss']

    # Apply deductible
    yelt_combined['NetLoss'] = (yelt_combined['NetLoss'] - 20e3).clip(lower=0.0)

    # Apply a limit
    yelt_combined['NetLoss'] = yelt_combined['NetLoss'].clip(upper=120e3)

    # Write to file
    yelt_combined.to_csv(ofilename2, index=True)
    print(f"Written to {ofilename2}")


if __name__ == "__main__":
    main()
