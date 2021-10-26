import pandas as pd
import matplotlib.pyplot as plt

from utils import smooth

def climateHockey():
  # download the data from https://ourworldindata.org/explorers/climate-change
  # get data
  try:
    df = pd.read_csv('climate-change.csv', parse_dates=["Day"])
  except FileNotFoundError as e:
    print("Download the data from https://ourworldindata.org/explorers/climate-change and save the csv in the current directory")
    return
  temp = df[df["Entity"] == "World"].set_index("Day")["temperature_anomaly"]
  # calculate smoothing
  y = smooth(temp)
  smoothed = pd.Series(y,index=temp.index)
  # xticks for plotting (data starts at 1880-01-15)
  idx = pd.date_range(start = temp.index[0], end = temp.index[-1], freq = "5Y") + pd.DateOffset(days=15,years=-1)
  # plot
  temp.plot(rot=45, xticks=idx, alpha=0.5)
  smoothed.plot(color='red') # for some reason also changes xticks to year-only
  plt.grid()
  plt.title("Temperature as deviation from the 1951-1980 mean")
  plt.ylabel("Temperature anomaly")
  plt.margins(0.01) # axis limits closer to graph
  plt.show()
