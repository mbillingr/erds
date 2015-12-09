![Python 2.7](https://img.shields.io/badge/python-2.7-green.svg)
![Python 3.5](https://img.shields.io/badge/python-3.5-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
ERDS
====
This package calculates and displays ERDS maps of event-related EEG/MEG data. ERDS is short for event-related desynchronization (ERD) and event-related synchronization (ERS). Conceptually, ERD corresponds to a decrease in power in a specific frequency band relative to a baseline. Similarly, ERS corresponds to an increase in power.

Usage
-----
The erds package uses an API similar to the one used in scikit-learn. Here is a simple example demonstrating the basic usage (note that the actual code for loading the data is missing):

    from erds import Erds

    maps = Erds()
    maps.fit(data)  # data must be available in appropriate format
    maps.plot()

Examples
--------
Example scripts demonstrating some features of the package can be found in the `examples` folder.

Dependencies
------------
The package depends on [NumPy](http://www.numpy.org/) and [matplotlib](http://matplotlib.org/).

References
----------
[G. Pfurtscheller, F. H. Lopes da Silva. Event-related EEG/MEG synchronization and desynchronization: basic principles. Clinical Neurophysiology 110(11), 1842-1857, 1999.][1]

[B. Graimann, J. E. Huggins, S. P. Levine, G. Pfurtscheller. Visualization of significant ERD/ERS patterns in multichannel EEG and ECoG data. Clinical  Neurophysiology 113(1), 43-47, 2002.][2]

[1]: http://dx.doi.org/10.1016/S1388-2457(99)00141-8
[2]: http://dx.doi.org/10.1016/S1388-2457(01)00697-6
