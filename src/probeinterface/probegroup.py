import numpy as np
from .utils import generate_unique_ids
from .probe import Probe


class ProbeGroup:
    """
    Class to handle a group of Probe objects and the global wiring to a device.

    Internally, this is represented as a list of Probe object.

    The ProbeGroup is the object saved in the json based probeinterface format, even if there is only one probe.

    Tiny detail: when using `PropbeGroup.to_numpy()` / `PropbeGroup.to_dataframe()` by default the contact order
    is the "natural" one (stacked order of each probe). An external contact order can be applied using the 
    ``ProbeGroup.set_global_contact_order()`` method, and the contact order is then stored in the 
    ``ProbeGroup._global_contact_order`` attribute. In this case, the contact order of the ProbeGroup is not "natural" 
    anymore, but the one defined by the user. This is useful for instance when some contact of each probe are 
    interleaved in the recording file.
    """

    def __init__(self):
        self.probes = []
        self.probe_ids = []
        self._global_contact_order = None

    def __repr__(self):
        repr_str = f"ProbeGroup: {len(self.probes)} probes - {self.get_contact_count()} contacts"
        if self._global_contact_order is not None:
            repr_str += " (with custom global contact order)"
        for probe, probe_id in zip(self.probes, self.probe_ids):
            repr_str += f"\n\t{probe_id}: {probe}"
        return repr_str

    def add_probe(self, probe: Probe, probe_id: str = None) -> None:
        """
        Add an additional probe to the ProbeGroup

        Parameters
        ----------
        probe: Probe
            The probe to add to the ProbeGroup
        probe_id: str, optional
            The ID to assign to the probe. If None, a unique ID will be generated.

        """
        if len(self.probes) > 0:
            self._check_compatible(probe)

        self.probes.append(probe)
        if probe_id is not None:
            self.probe_ids.append(probe_id)
        else:
            self.probe_ids.append(f"probe_{len(self.probes)}")        
        probe._probe_group = self

    def set_probe_ids(self, probe_ids: list) -> None:
        """
        Set the probe IDs for the ProbeGroup.

        Parameters
        ----------
        probe_ids: list
            A list of IDs to assign to the probes. 
            The length of the list must match the number of probes in the ProbeGroup.
        """
        if len(probe_ids) != len(self.probes):
            raise ValueError(
                f"Length of probe_ids ({len(probe_ids)}) does not match number of probes ({len(self.probes)})"
            )
        self.probe_ids = probe_ids

    def _check_compatible(self, probe: Probe) -> None:
        if probe._probe_group is not None:
            raise ValueError(
                "This probe is already attached to another ProbeGroup. Use probe.copy() to attach it to another ProbeGroup"
            )

        if probe.ndim != self.probes[-1].ndim:
            raise ValueError(
                f"ndim are not compatible: probe.ndim {probe.ndim} " f"!= probegroup ndim {self.probes[-1].ndim}"
            )

        # check global channel maps
        self.probes.append(probe)
        self.check_global_device_wiring_and_ids()
        self.probes = self.probes[:-1]

    @property
    def ndim(self) -> int:
        return self.probes[0].ndim

    def copy(self) -> "ProbeGroup":
        """
        Create a copy of the ProbeGroup

        Returns
        -------
        copy: ProbeGroup
            A copy of the ProbeGroup
        """
        return ProbeGroup.from_dict(self.to_dict(array_as_list=False))

    def get_contact_count(self) -> int:
        """
        Total number of channels.

        Returns
        -------
        n: int
            The total number of channels
        """
        n = sum(probe.get_contact_count() for probe in self.probes)
        return n

    def to_numpy(self, complete: bool = False) -> np.ndarray:
        """
        Export all probes into a numpy array.

        Parameters
        ----------
        complete: bool, default: False
            If True, export complete information about the probegroup
            including contact_plane_axes/si_units/device_channel_indices
        """

        fields = []
        probe_arr = []

        # loop over probes to get all fields
        dtype = [("probe_index", "int64")]
        fields = []
        for probe_index, probe in enumerate(self.probes):
            arr = probe.to_numpy(complete=complete)
            probe_arr.append(arr)
            for k in arr.dtype.fields:
                if k not in fields:
                    fields.append(k)
                    dtype += [(k, arr.dtype.fields[k][0])]

        pg_arr = []
        for probe_index, probe in enumerate(self.probes):
            arr = probe_arr[probe_index]
            arr_ext = np.zeros(probe.get_contact_count(), dtype=dtype)
            arr_ext["probe_index"] = probe_index
            for k in fields:
                if k in arr.dtype.fields:
                    arr_ext[k] = arr[k]
            pg_arr.append(arr_ext)

        pg_arr = np.concatenate(pg_arr, axis=0)

        if self._global_contact_order is not None:
            pg_arr = pg_arr[self._global_contact_order]
        return pg_arr

    @staticmethod
    def from_numpy(arr: np.ndarray) -> "ProbeGroup":
        """Create ProbeGroup from a complex numpy array
        see ProbeGroup.to_numpy()

        Note that if the contact_vector has several probe and some contact are interleaved, then the ProbeGroup will
        have a non natural ordering (contact from probes are stack): in short ProbeGroup._global_contact_order
        will be not None.

        Parameters
        ----------
        arr : np.array
            The structured np.array representation of the probe

        Returns
        -------
        probegroup : ProbeGroup
            The instantiated ProbeGroup object
        """

        # Check if contacts are interleaved
        num_probes = np.unique(arr["probe_index"]).size
        is_interleaved = (num_probes > 1) and np.any(np.diff(arr["probe_index"]) < 0)
        if is_interleaved:
            global_contact_order = []

        probes_indices = np.unique(arr["probe_index"])
        probegroup = ProbeGroup()
        for probe_index in probes_indices:
            mask = arr["probe_index"] == probe_index
            probe = Probe.from_numpy(arr[mask])
            probegroup.add_probe(probe)

            if is_interleaved:
                global_contact_order.append(np.flatnonzero(mask))

        if is_interleaved:
            # the argsort is for the 'reverse' order!
            probegroup._global_contact_order = np.argsort(np.concatenate(global_contact_order))

        return probegroup

    def to_dataframe(self, complete: bool = False) -> "pandas.DataFrame":
        """
        Export the probegroup to a pandas dataframe

        Parameters
        ----------
        complete : bool, default: False
            If True, export complete information about the probegroup,
            including the probe plane axis.

        Returns
        -------
        df : pandas.DataFrame
            The dataframe representation of the probegroup

        """
        import pandas as pd

        df = pd.DataFrame(self.to_numpy(complete=complete))
        df.index = np.arange(df.shape[0], dtype="int64")
        return df

    def to_dict(self, array_as_list: bool = False) -> dict:
        """Create a dictionary of all necessary attributes.

        Parameters
        ----------
        array_as_list : bool, default: False
            If True, arrays are converted to lists, by default False

        Returns
        -------
        d : dict
            The dictionary representation of the probegroup
        """
        d = {}
        d["probes"] = []
        for probe in self.probes:
            probe_dict = probe.to_dict(array_as_list=array_as_list)
            d["probes"].append(probe_dict)
        if self._global_contact_order is not None:
            global_contact_order = self._global_contact_order
            if array_as_list:
                global_contact_order = global_contact_order.tolist()
            d["global_contact_order"] = global_contact_order
        return d

    @staticmethod
    def from_dict(d: dict) -> "ProbeGroup":
        """Instantiate a ProbeGroup from a dictionary

        Parameters
        ----------
        d : dict
            The dictionary representation of the probegroup

        Returns
        -------
        probegroup : ProbeGroup
            The instantiated ProbeGroup object
        """
        probegroup = ProbeGroup()
        for probe_dict in d["probes"]:
            probe = Probe.from_dict(probe_dict)
            probegroup.add_probe(probe)

        global_contact_order = d.get("global_contact_order", None)
        if global_contact_order is not None:
            probegroup._global_contact_order = np.asarray(global_contact_order)

        return probegroup

    # TODO: this should only return the device_channel_indices, not the probe_index!!!
    def get_global_device_channel_indices(self) -> np.ndarray:
        """
        Gets the global device channels indices and returns as
        an array

        Returns
        -------
        channels: np.ndarray
            a numpy array vector with 2 columns
            (probe_index, device_channel_indices)

        Notes
        -----
            If a channel within channels has a value of -1 this indicates that that channel
            is disconnected
        """
        total_chan = self.get_contact_count()
        channels = np.zeros(total_chan, dtype=[("probe_index", "int64"), ("device_channel_indices", "int64")])
        arr = self.to_numpy(complete=True)
        channels["probe_index"] = arr["probe_index"]
        channels["device_channel_indices"] = arr["device_channel_indices"]
        return channels

    def set_global_device_channel_indices(self, device_channel_indices: np.ndarray | list) -> None:
        """
        Set global device channel indices for all probes.

        Important note: if the probegroup has ``_global_contact_order``, then the device_channel_indices
        are reordered before being set. In short, the ``device_channel_indices`` is zipped to
        ProbeGroup.to_numpy() (always ordered).

        Parameters
        ----------
        device_channel_indices: np.ndarray | list
            The device channal indices to be set
        """
        device_channel_indices = np.asarray(device_channel_indices)
        if device_channel_indices.size != self.get_contact_count():
            raise ValueError(
                f"Wrong channels size {device_channel_indices.size} for the number of channels {self.get_contact_count()}"
            )

        # first reset previous indices
        for i, probe in enumerate(self.probes):
            n = probe.get_contact_count()
            probe.set_device_channel_indices([-1] * n)

        if self._global_contact_order is not None:
            # this is tricky conceptually but needed for consistency
            rev_order = np.argsort(self._global_contact_order)
            device_channel_indices = device_channel_indices[rev_order]

        # then set new indices
        ind = 0
        for i, probe in enumerate(self.probes):
            n = probe.get_contact_count()
            probe.set_device_channel_indices(device_channel_indices[ind : ind + n])
            ind += n

    def get_global_contact_ids(self) -> np.ndarray:
        """
        Gets all contact ids concatenated across probes

        Returns
        -------
        contact_ids: np.ndarray
            An array of the contact ids across all probes
        """
        contact_ids = self.to_numpy(complete=True)["contact_ids"]
        return contact_ids

    def get_global_contact_positions(self) -> np.ndarray:
        """
        Gets all contact positions concatenated across probes

        Returns
        -------
        contact_positions: np.ndarray
            An array of the contact positions across all probes
        """
        contact_positions = np.vstack([probe.contact_positions for probe in self.probes])
        if self._global_contact_order is not None:
            contact_positions = contact_positions[self._global_contact_order]
        return contact_positions

    def get_slice(self, selection: np.ndarray[bool | int]) -> "ProbeGroup":
        """
        Get a copy of the ProbeGroup with a sub selection of contacts.

        Selection can be boolean or by index

        Parameters
        ----------
        selection : np.array of bool or int (for index)
            Either an np.array of bool or for desired selection of contacts
            or the indices of the desired contacts

        Returns
        -------
        sliced_probe_group: ProbeGroup
            The sliced probe group

        """

        n = self.get_contact_count()

        selection = np.asarray(selection)
        if selection.dtype.kind not in ("b", "i"):
            raise TypeError(f"selection must be bool array or int array, not of type: {type(selection)}")

        if selection.dtype == "bool":
            assert selection.shape == (
                n,
            ), f"if array of bool given it must be the same size as the number of contacts {selection.shape} != {n}"
            selection = np.flatnonzero(selection)

        if len(selection) == 0:
            raise ValueError("ProbeGroup.get_slice() with empty selection is not handled")
            # return ProbeGroup()

        assert np.unique(selection).size == selection.size
        assert 0 <= np.min(selection) < n, f"An index within your selection is out of bounds {np.min(selection)}"
        assert 0 <= np.max(selection) < n, f"An index within your selection is out of bounds {np.max(selection)}"

        contact_arr = self.to_numpy(complete=True)
        contact_arr = contact_arr[selection]
        original_probe_indices = np.unique(contact_arr["probe_index"])
        sliced_probe_group = ProbeGroup.from_numpy(contact_arr)
        new_probe_indices = np.unique(sliced_probe_group.to_numpy(complete=True)["probe_index"])

        # Map annotations of the original probegroup to the sliced one
        new_probe_ids = [self.probe_ids[i] for i in original_probe_indices]
        sliced_probe_group.set_probe_ids(new_probe_ids)
        for original_probe_index, new_probe_index in zip(original_probe_indices, new_probe_indices):
            orig_probe = self.probes[original_probe_index]
            new_probe = sliced_probe_group.probes[new_probe_index]
            
            for k in orig_probe.annotations:
                if k not in new_probe.annotations:
                    new_probe.annotate(**{k: orig_probe.annotations[k]})

        return sliced_probe_group

    def select_contacts(self, contact_ids: np.ndarray | list | None = None, probe_ids: np.ndarray | list | None = None) -> "ProbeGroup":
        """
        Get a copy of the ProbeGroup with a sub selection of contacts based on contact ids and probe ids.

        Parameters
        ----------
        contact_ids : np.array or list or None, default: None
            The contact ids to select. If None, all contacts are selected, but probe_ids must be provided.
        probe_ids : np.array or list or None, default: None
            The probe ids to select. If contact_ids are not unique across probes, 
            then probe_ids should be provided to disambiguate. 
            If contact_ids are unique across probes, then probe_ids can be None.

        Returns
        -------
        sliced_probe_group: ProbeGroup
            The sliced probe group
        """
        if contact_ids is None and probe_ids is None:
            raise ValueError(
                "Either contact_ids or probe_ids must be provided for selection."
            )
        if contact_ids is None:
            contact_mask = np.ones(self.get_contact_count(), dtype=bool)
        else:
            contact_ids = np.asarray(contact_ids)
            all_contact_ids = self.get_global_contact_ids()
            contact_mask = np.isin(all_contact_ids, contact_ids)
            if probe_ids is None:
                # without probe_ids the selection must be unambiguous: every requested
                # contact id must match a single contact across the whole ProbeGroup
                matched_ids = all_contact_ids[contact_mask]
                unique_ids, counts = np.unique(matched_ids, return_counts=True)
                ambiguous_ids = unique_ids[counts > 1]
                if ambiguous_ids.size > 0:
                    raise ValueError(
                        f"contact_ids {ambiguous_ids.tolist()} are not unique across probes, "
                        "you should provide probe_ids to disambiguate"
                    )
        if probe_ids is None:
            probe_mask = np.ones(self.get_contact_count(), dtype=bool)
        else:
            all_probe_ids = np.asarray(self.probe_ids)[self.to_numpy(complete=True)["probe_index"]]
            probe_ids = np.asarray(probe_ids)
            probe_mask = np.isin(all_probe_ids, probe_ids)
        selection_mask = contact_mask & probe_mask
        return self.get_slice(selection_mask)

    def set_global_contact_order(self, global_contact_order: np.ndarray | list) -> None:
        """
        Set the global contact order for the ProbeGroup. This is useful when some contact of each probe are interleaved in the recording file.

        Parameters
        ----------
        global_contact_order: np.ndarray | list
            The global contact order to be set. It should be an array of indices that defines the new order of contacts across all probes.
        """
        global_contact_order = np.asarray(global_contact_order)
        if global_contact_order.size != self.get_contact_count():
            raise ValueError(
                f"Wrong global contact order size {global_contact_order.size} for the number of channels {self.get_contact_count()}"
            )
        self._global_contact_order = global_contact_order

    def check_global_device_wiring_and_ids(self) -> None:
        # check unique device_channel_indices for !=-1
        chans = self.get_global_device_channel_indices()
        keep = chans["device_channel_indices"] >= 0
        valid_chans = chans[keep]["device_channel_indices"]

        if valid_chans.size != np.unique(valid_chans).size:
            raise ValueError("channel device indices are not unique across probes")

    def auto_generate_probe_ids(self, *args, **kwargs) -> None:
        """
        Annotate all probes with unique probe_id values.

        Parameters
        ----------
        *args: will be forwarded to `probeinterface.utils.generate_unique_ids`
        **kwargs: will be forwarded to
            `probeinterface.utils.generate_unique_ids`
        """

        if any("probe_id" in p.annotations for p in self.probes):
            raise ValueError("Probe already has a `probe_id` annotation.")

        if not args:
            args = 1e7, 1e8
        # 3rd argument has to be the number of probes
        args = args[:2] + (len(self.probes),)

        # creating unique probe ids in case probes do not have any yet
        probe_ids = generate_unique_ids(*args, **kwargs).astype(str)
        for pid, probe in enumerate(self.probes):
            probe.annotate(probe_id=probe_ids[pid])

    def auto_generate_contact_ids(self, *args, **kwargs) -> None:
        """
        Annotate all contacts with unique contact_id values.

        Parameters
        ----------
        *args: will be forwarded to `probeinterface.utils.generate_unique_ids`
        **kwargs: will be forwarded to
            `probeinterface.utils.generate_unique_ids`
        """

        if not args:
            args = 1e7, 1e8
        # 3rd argument has to be the number of probes
        args = args[:2] + (self.get_contact_count(),)

        contact_ids = generate_unique_ids(*args, **kwargs).astype(str)

        for probe in self.probes:
            el_ids, contact_ids = np.split(contact_ids, [probe.get_contact_count()])
            probe.set_contact_ids(el_ids)
