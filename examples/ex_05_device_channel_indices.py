"""
Handle channel indices
----------------------

Probes can have a complex contacts indexing system due to the probe layout.
When they are plugged into a recording device like an Open Ephys with an Intan headstage,
the channel order can be mixed again. So the physical contact channel index
is rarely the channel index on the device.

This is why the `Probe` object can handle separate `device_channel_indices`.
"""

##############################################################################
# Import

import numpy as np
import matplotlib.pyplot as plt

from probeinterface import Probe, ProbeGroup
from probeinterface.plotting import plot_probe, plot_probegroup
from probeinterface import generate_multi_columns_probe

##############################################################################
# Let's first generate a probe. By default, the wiring is not complicated:
# each column increments the contact index from the bottom to the top of the probe:

probe = generate_multi_columns_probe(num_columns=3,
                                     num_contact_per_column=[5, 6, 5],
                                     xpitch=75, ypitch=75, y_shift_per_column=[0, -37.5, 0],
                                     contact_shapes='circle', contact_shape_params={'radius': 12})

plot_probe(probe, with_contact_id=True)

##############################################################################
# The Probe is not connected to any device yet:

print(probe.device_channel_indices)

##############################################################################
# Let's imagine we have a headstage with the following wiring: the first half
# of the channels have natural indices, but the order of the other half is reversed:

channel_indices = np.arange(16)
channel_indices[8:16] = channel_indices[8:16][::-1]
probe.set_device_channel_indices(channel_indices)
print(probe.device_channel_indices)

##############################################################################
#  We can visualize the two sets of indices:
#  
#  * the prbXX is the contact index ordered from 0 to N
#  * the devXX is the channel index on the device (with the second half reversed)

plot_probe(probe, with_contact_id=True, with_device_index=True)

##############################################################################
# Very often we have several probes on the device and this can lead to even
# more complex channel indices.
# `ProbeGroup.get_global_device_channel_indices()` gives an overview of the device wiring.

probe0 = generate_multi_columns_probe(num_columns=3,
                                      num_contact_per_column=[5, 6, 5],
                                      xpitch=75, ypitch=75, y_shift_per_column=[0, -37.5, 0],
                                      contact_shapes='circle', contact_shape_params={'radius': 12})
probe1 = probe0.copy()

probe1.move([350, 200])
probegroup = ProbeGroup()
probegroup.add_probe(probe0)
probegroup.add_probe(probe1)

# wire probe0 0 to 31 and shuffle
channel_indices0 = np.arange(16)
np.random.shuffle(channel_indices0)
probe0.set_device_channel_indices(channel_indices0)

# wire probe0 32 to 63 and shuffle
channel_indices1 = np.arange(16, 32)
np.random.shuffle(channel_indices1)
probe1.set_device_channel_indices(channel_indices1)

print(probegroup.get_global_device_channel_indices())

##############################################################################
# The indices of the probe group can also be plotted:

fig, ax = plt.subplots()
plot_probegroup(probegroup, with_contact_id=True, same_axes=True, ax=ax)

##############################################################################
# Reordering contacts with a global contact order
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# By default the contact order of a `ProbeGroup` is the "natural" one: the
# contacts of each probe are stacked one probe after the other. But sometimes
# the contacts of the different probes are *interleaved* in the recording file
# (e.g. the acquisition system alternates between probes sample by sample).
#
# `ProbeGroup.set_global_contact_order()` lets us store this external ordering.
# The order is an array of indices into the natural (stacked) order, and it is
# applied whenever the group is exported with `to_numpy()` / `to_dataframe()`.

probegroup = ProbeGroup()
probegroup.add_probe(probe0.copy())
probegroup.add_probe(probe1.copy())

n = probegroup.get_contact_count()
print("default global contact order:", probegroup._global_contact_order)

# interleave probe0 and probe1 contacts as they appear in the recording file
global_contact_order = np.zeros(n, dtype="int64")
global_contact_order[0::2] = np.arange(0, n // 2)        # probe0 contacts
global_contact_order[1::2] = np.arange(n // 2, n)        # probe1 contacts
probegroup.set_global_contact_order(global_contact_order)

##############################################################################
# Now `to_numpy()` returns the contacts in the interleaved order: the
# ``probe_index`` column alternates between the two probes.

contact_vector = probegroup.to_numpy()
print("probe_index in global order:", contact_vector["probe_index"][:8])

##############################################################################
# The global order interacts with `set_global_device_channel_indices()`: the
# ``device_channel_indices`` you pass are interpreted in the (reordered) order
# returned by `to_numpy()`, so they map directly onto the acquisition channels.

probegroup.set_global_device_channel_indices(np.arange(n))
print("device_channel_indices (global order):",
      probegroup.to_numpy(complete=True)["device_channel_indices"][:8])

plt.show()
