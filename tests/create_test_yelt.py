"""Generate a YELT for the tests"""
import pandas as pd
import numpy as np


def main():
    """Main script"""

    seed = 12345
    n_years = 1e5  # Number of years
    freq0 = 0.5  # Number of events per year
    alpha = 1.0
    min_loss = 1e3
    ofilename = "_data/example_pareto_poisson_yelt.csv"

    rs = np.random.RandomState(seed=seed)

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
        'DayOfYear': 1 + (365 * np.mod(event_times, 1.0)),
        'Loss': losses,
    })

    # Sort based on time
    yelt = yelt.sort_values(['Year', 'DayOfYear', 'EventID', 'Loss'])

    # Convert to a series
    yelt = yelt.set_index(['Year', 'EventID', 'DayOfYear'],
                          verify_integrity=True)

    # Write to file
    yelt.to_csv(ofilename, index=True)
    print(f"Written to {ofilename}")


if __name__ == "__main__":
    main()
