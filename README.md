![Python 2.7](https://img.shields.io/badge/python-2.7-green.svg)
![Python 3.5](https://img.shields.io/badge/python-3.5-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
ERDS
====
This package calculates and displays ERDS maps of event-related EEG/MEG data. ERDS is short for event-related desynchronization (ERD) and event-related synchronization (ERS). Conceptually, ERD corresponds to a decrease in power in a specific frequency band relative to a baseline. Similarly, ERS corresponds to an increase in power.

Usage
-----
The erds package uses a similar API like scikit-learn. Here is a simple example demonstrating the basic usage (note that the actual code for loading the data is missing):

    from erds import Erds

    maps = Erds()
    maps.fit(data)  # data must be available in appropriate format
    maps.plot()

Dependencies
------------
The package depends on [NumPy](http://www.numpy.org/) and [matplotlib](http://matplotlib.org/).
