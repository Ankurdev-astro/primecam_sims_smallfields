# Copyright (c) 2026
# Standalone TOAST-style Lissajous ground-scan operator.
#
# This is intentionally not a drop-in replacement for toast.ops.SimGround.
# It keeps the observation / telescope / weather / pointing construction pattern
# used by SimGround, but replaces the constant-elevation CES generator with a
# smooth Lissajous AZ/EL trajectory.

import copy
from datetime import datetime, timezone

import numpy as np
import traitlets
from astropy import units as u
from astropy.table import QTable

from toast import qarray as qa
from toast.coordinates import azel_to_radec
from toast.dist import distribute_discrete
from toast.instrument import Focalplane, Session, Telescope
from toast.intervals import IntervalList
from toast.io import load_instrument_file
from toast.observation import Observation
from toast.observation import default_values as defaults
from toast.schedule import GroundSchedule
from toast.timing import function_timer
from toast.traits import (
    Bool,
    Instance,
    Int,
    Quantity,
    Unicode,
    Unit,
    trait_docs,
)
from toast.utils import Logger, memreport, name_UID
from toast.weather import SimWeather
from toast.ops.operator import Operator
from toast.ops.sim_ground_utils import add_solar_intervals


__all__ = ["SimGroundLissajous"]


@trait_docs
class SimGroundLissajous(Operator):
    """Simulate a ground-based Lissajous boresight scan.

    The operator follows the broad structure of ``toast.ops.SimGround``:
    it consumes a ``GroundSchedule`` and a ``Telescope``, creates TOAST
    observations, writes timestamp / AZ / EL / boresight quaternion shared
    fields, and records Sun-related intervals.

    The scan model is

        az(t) = az_center + az_drift * t + A_az sin(2 pi f_az t + phi_az)
        el(t) = el_center + el_drift * t + A_el sin(2 pi f_el t + phi_el)

    where the default AZ center is the midpoint of ``scan.az_min`` and
    ``scan.az_max`` and the default EL center is ``scan.el``.  If
    ``lissajous_az_amplitude`` is left as ``None``, half of the scheduled
    AZ throw is used.

    No physical turnaround intervals are generated.  For compatibility with
    TOAST operators that expect the standard interval key, an empty turnaround
    interval list is still created.  The full observation is represented as
    one continuous scanning interval.

    Example
    -------
    sim = SimGroundLissajous(weather="atacama")
    sim.telescope = telescope
    sim.schedule = schedule
    sim.lissajous_az_rate = 0.05 * u.Hz
    sim.lissajous_el_rate = 0.071 * u.Hz
    sim.lissajous_az_amplitude = 1.0 * u.deg
    sim.lissajous_el_amplitude = 0.6 * u.deg
    sim.apply(data)
    """

    API = Int(0, help="Internal interface version for this operator")

    telescope = Instance(
        klass=Telescope, allow_none=True, help="Telescope instance with a focalplane"
    )

    telescope_file = Unicode(
        None, allow_none=True, help="Path to HDF5 file containing an instrument group"
    )

    session_split_key = Unicode(
        None,
        allow_none=True,
        help="Focalplane key used to split the telescope into observations",
    )

    weather = Unicode(
        None,
        allow_none=True,
        help="Name of built-in weather site or path to weather HDF5 file",
    )

    realization = Int(0, help="Weather realization index")

    schedule = Instance(
        klass=GroundSchedule, allow_none=True, help="Instance of a GroundSchedule"
    )

    schedule_file = Unicode(
        None, allow_none=True, help="Ground-based observing schedule file"
    )

    sort_schedule_file = Bool(
        False, help="If True, sort loaded schedule by mean boresight RA"
    )

    use_ephem = Bool(
        True, help="Use PyEphem to convert between horizontal and equatorial systems"
    )

    use_qpoint = Bool(
        False, help="Use qpoint to convert between horizontal and equatorial systems"
    )

    # Lissajous parameters
    lissajous_az_amplitude = Quantity(
        None,
        allow_none=True,
        help="AZ Lissajous amplitude. If None, use half the scheduled AZ throw.",
    )

    lissajous_el_amplitude = Quantity(
        1.0 * u.degree, help="EL Lissajous amplitude"
    )

    lissajous_az_rate = Quantity(
        0.05 * u.Hz, help="AZ Lissajous oscillation frequency"
    )

    lissajous_el_rate = Quantity(
        0.071 * u.Hz, help="EL Lissajous oscillation frequency"
    )

    lissajous_az_phase = Quantity(
        0.0 * u.degree, help="AZ Lissajous phase offset"
    )

    lissajous_el_phase = Quantity(
        90.0 * u.degree, help="EL Lissajous phase offset"
    )

    lissajous_az_drift_rate = Quantity(
        0.0 * u.degree / u.second, help="Linear AZ center drift rate"
    )

    lissajous_el_drift_rate = Quantity(
        0.0 * u.degree / u.second, help="Linear EL center drift rate"
    )

    check_mount_motion_limits = Bool(
        True,
        help=(
            "Check Lissajous peak mount velocity / acceleration against "
            "the max_mount_rate and max_mount_accel traits"
        ),
    )

    # These are mount motion limits for the Lissajous pattern, not absolute
    # mechanical AZ / EL pointing bounds.
    max_mount_rate_az = Quantity(
        1.0 * u.degree / u.second, help="Maximum allowed AZ mount rate"
    )

    max_mount_rate_el = Quantity(
        1.0 * u.degree / u.second, allow_none=True, help="Maximum allowed EL mount rate"
    )

    max_mount_accel_az = Quantity(
        1.0 * u.degree / u.second**2, help="Maximum allowed AZ mount acceleration"
    )

    max_mount_accel_el = Quantity(
        1.0 * u.degree / u.second**2,
        allow_none=True,
        help="Maximum allowed EL mount acceleration",
    )

    distribute_time = Bool(
        False, help="Distribute observation data along time rather than detector axis"
    )

    sample_set_size = Int(
        0,
        help=(
            "Samples per sample-set when distributing by time. "
            "If zero, split into at least one sample-set per sample rank."
        ),
    )

    detset_key = Unicode(
        None,
        allow_none=True,
        help="Focalplane column used to group detectors into detector sets",
    )

    times = Unicode(defaults.times, help="Observation shared key for timestamps")

    shared_flags = Unicode(
        defaults.shared_flags,
        allow_none=True,
        help="Observation shared key for common flags; created as zeros if not None",
    )

    det_data = Unicode(
        defaults.det_data,
        allow_none=True,
        help="Observation detdata key to initialize",
    )

    det_data_units = Unit(
        defaults.det_data_units, help="Output units if creating detector data"
    )

    det_flags = Unicode(
        defaults.det_flags,
        allow_none=True,
        help="Observation detdata key for flags to initialize",
    )

    azimuth = Unicode(defaults.azimuth, help="Observation shared key for Azimuth")

    elevation = Unicode(defaults.elevation, help="Observation shared key for Elevation")

    boresight_azel = Unicode(
        defaults.boresight_azel, help="Observation shared key for boresight AZ/EL"
    )

    boresight_radec = Unicode(
        defaults.boresight_radec, help="Observation shared key for boresight RA/DEC"
    )

    position = Unicode(defaults.position, help="Observation shared key for position")

    velocity = Unicode(defaults.velocity, help="Observation shared key for velocity")

    scanning_interval = Unicode(
        defaults.scanning_interval, help="Interval name for the full Lissajous scan"
    )

    turnaround_interval = Unicode(
        defaults.turnaround_interval,
        allow_none=True,
        help="Interval name for turnarounds. Created empty for Lissajous scans.",
    )

    sun_up_interval = Unicode(
        defaults.sun_up_interval, help="Interval name for times when the Sun is up"
    )

    sun_close_interval = Unicode(
        defaults.sun_close_interval, help="Interval name for times when the Sun is close"
    )

    sun_close_distance = Quantity(45.0 * u.degree, help="Sun-close flagging distance")

    max_pwv = Quantity(
        None, allow_none=True, help="Maximum PWV for the simulated weather"
    )

    median_weather = Bool(
        False, help="Use median weather parameters instead of sampling distributions"
    )

    @traitlets.validate("telescope")
    def _check_telescope(self, proposal):
        tele = proposal["value"]
        if tele is not None:
            try:
                _ = tele.focalplane.detectors
            except Exception as exc:
                raise traitlets.TraitError(
                    "telescope must be a Telescope instance with a focalplane"
                ) from exc
        return tele

    @traitlets.validate("use_ephem")
    def _check_use_ephem(self, proposal):
        use_ephem = proposal["value"]
        if use_ephem and self.use_qpoint:
            raise traitlets.TraitError("Cannot use both ephem and qpoint")
        return use_ephem

    @traitlets.validate("use_qpoint")
    def _check_use_qpoint(self, proposal):
        use_qpoint = proposal["value"]
        if use_qpoint and self.use_ephem:
            raise traitlets.TraitError("Cannot use both ephem and qpoint")
        return use_qpoint

    @traitlets.validate("schedule")
    def _check_schedule(self, proposal):
        sch = proposal["value"]
        if sch is not None and not isinstance(sch, GroundSchedule):
            raise traitlets.TraitError("schedule must be an instance of GroundSchedule")
        return sch

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()
        comm = data.comm

        det_ranks = comm.group_size
        samp_ranks = 1
        if self.distribute_time:
            det_ranks = 1
            samp_ranks = comm.group_size

        if self.schedule is None and self.schedule_file is None:
            raise RuntimeError(
                "Either the schedule or schedule_file must be set before calling exec()"
            )

        if self.telescope is None and self.telescope_file is None:
            raise RuntimeError(
                "Either the telescope or telescope_file must be specified"
            )

        if self.telescope is None:
            if comm.world_rank == 0:
                self.telescope, _ = load_instrument_file(self.telescope_file)
            if comm.comm_world is not None:
                self.telescope = comm.comm_world.bcast(self.telescope, root=0)

        if self.schedule is None:
            schedule = GroundSchedule()
            schedule.read(self.schedule_file, comm=comm.comm_world)
            if self.sort_schedule_file:
                schedule.sort_by_RA()
        else:
            schedule = self.schedule

        if len(schedule.scans) == 0:
            raise RuntimeError("Schedule has no scans")

        obs_tele = self._obs_telescopes(data, det_ranks, detectors)

        mission_start = schedule.scans[0].start
        rate = self.telescope.focalplane.sample_rate.to_value(u.Hz)
        incr = 1.0 / rate

        obs_info = []
        off = 0
        for scan in schedule.scans:
            length = (scan.stop - scan.start).total_seconds()
            if np.abs(length) < incr:
                continue

            ffirst = rate * (scan.start - mission_start).total_seconds()
            first = int(ffirst)
            if ffirst - first > 1.0e-3 * incr:
                first += 1
            start = first * incr + mission_start.timestamp()
            nsamp = 1 + int(rate * (scan.stop.timestamp() - start))

            sname = f"{scan.name}-{scan.scan_indx}-{scan.subscan_indx}"
            for obkey, (obtele, detsets) in obs_tele.items():
                obs_name = sname if obkey == "ALL" else f"{sname}_{obkey}"
                obs_info.append(
                    {
                        "name": obs_name,
                        "sname": sname,
                        "obkey": obkey,
                        "scan": scan,
                        "start": start,
                        "samples": nsamp,
                        "offset": off,
                    }
                )
            off += nsamp

        if len(obs_info) == 0:
            raise RuntimeError("Schedule produced no valid observations")

        obs_samples = [x["samples"] for x in obs_info]
        groupdist = distribute_discrete(obs_samples, comm.ngroups)
        group_first_obs = groupdist[comm.group][0]
        group_num_obs = groupdist[comm.group][1]

        last_session = None
        for obindx in range(group_first_obs, group_first_obs + group_num_obs):
            scan = obs_info[obindx]["scan"]
            sname = obs_info[obindx]["sname"]
            obs_name = obs_info[obindx]["name"]

            sys_mem_str = memreport(
                msg="(whole node)", comm=data.comm.comm_group, silent=True
            )
            log.debug_rank(
                f"Group {data.comm.group} begin observation {obs_name} with {sys_mem_str}",
                comm=data.comm.comm_group,
            )

            if sname != last_session:
                site = self.telescope.site
                (
                    times,
                    az,
                    el,
                    sample_sets,
                    scan_min_az,
                    scan_max_az,
                    scan_min_el,
                    scan_max_el,
                    scan_interval,
                ) = self._simulate_scanning(
                    scan, obs_info[obindx]["samples"], rate, comm, samp_ranks
                )

                if self.weather is not None:
                    site = copy.deepcopy(site)
                    mid_time = scan.start + (scan.stop - scan.start) / 2
                    try:
                        weather = SimWeather(
                            time=mid_time,
                            name=self.weather,
                            site_uid=site.uid,
                            realization=self.realization,
                            max_pwv=self.max_pwv,
                            median_weather=self.median_weather,
                        )
                    except RuntimeError:
                        weather = SimWeather(
                            time=mid_time,
                            file=self.weather,
                            site_uid=site.uid,
                            realization=self.realization,
                            max_pwv=self.max_pwv,
                            median_weather=self.median_weather,
                        )
                    site.weather = weather

                session = Session(
                    sname,
                    start=datetime.fromtimestamp(times[0]).astimezone(timezone.utc),
                    end=datetime.fromtimestamp(times[-1]).astimezone(timezone.utc),
                )

            obtele, detsets = obs_tele[obs_info[obindx]["obkey"]]
            telescope = Telescope(
                obtele.name,
                uid=obtele.uid,
                focalplane=obtele.focalplane,
                site=site,
            )

            ob = Observation(
                comm,
                telescope,
                len(times),
                name=obs_name,
                uid=name_UID(obs_name),
                session=session,
                detector_sets=detsets,
                process_rows=det_ranks,
                sample_sets=sample_sets,
            )

            ob["scan_el"] = scan.el
            ob["scan_min_az"] = scan_min_az * u.radian
            ob["scan_max_az"] = scan_max_az * u.radian
            ob["scan_min_el"] = scan_min_el * u.radian
            ob["scan_max_el"] = scan_max_el * u.radian
            ob["scan_type"] = "lissajous"

            self._create_shared_fields(ob)

            if self.det_data is not None:
                ob.detdata.ensure(
                    self.det_data,
                    dtype=np.float64,
                    create_units=self.det_data_units,
                )

            if self.det_flags is not None:
                ob.detdata.ensure(self.det_flags, dtype=np.uint8)

            stamps = None
            position = None
            velocity = None
            az_data = None
            el_data = None
            bore_azel = None
            bore_radec = None

            if ob.comm_col_rank == 0:
                i0 = ob.local_index_offset
                i1 = i0 + ob.n_local_samples
                stamps = times[i0:i1]
                az_data = az[i0:i1]
                el_data = el[i0:i1]

                position, velocity = site.position_velocity(stamps)

                bore_azel = qa.from_lonlat_angles(
                    -az_data, el_data, np.zeros_like(el_data)
                )

                if scan.boresight_angle.to_value(u.radian) != 0:
                    zaxis = np.array([0.0, 0.0, 1.0])
                    rot = qa.rotation(zaxis, scan.boresight_angle.to_value(u.radian))
                    bore_azel = qa.mult(bore_azel, rot)

                bore_radec = self._azel_to_radec(site, stamps, bore_azel)

            ob.shared[self.times].set(stamps, offset=(0,), fromrank=0)
            ob.shared[self.azimuth].set(az_data, offset=(0,), fromrank=0)
            ob.shared[self.elevation].set(el_data, offset=(0,), fromrank=0)
            ob.shared[self.position].set(position, offset=(0, 0), fromrank=0)
            ob.shared[self.velocity].set(velocity, offset=(0, 0), fromrank=0)
            ob.shared[self.boresight_azel].set(bore_azel, offset=(0, 0), fromrank=0)
            ob.shared[self.boresight_radec].set(bore_radec, offset=(0, 0), fromrank=0)

            if self.shared_flags is not None:
                ob.shared[self.shared_flags].data[:] = 0

            ob.intervals[self.scanning_interval] = IntervalList(
                ob.shared[self.times], timespans=[scan_interval]
            )

            if self.turnaround_interval is not None:
                ob.intervals[self.turnaround_interval] = IntervalList(
                    ob.shared[self.times], timespans=[]
                )

            add_solar_intervals(
                ob.intervals,
                site,
                ob.shared[self.times],
                ob.shared[self.azimuth].data,
                ob.shared[self.elevation].data,
                self.sun_up_interval,
                self.sun_close_interval,
                self.sun_close_distance,
            )

            log.verbose_rank(
                f"Group {data.comm.group} finished observation {obs_name}:\n{ob}",
                comm=data.comm.comm_group,
            )

            obmem_gb = ob.memory_use() / 1024**3
            log.debug_rank(
                f"Observation {ob.name} using {obmem_gb:0.2f} GB of total memory",
                comm=ob.comm.comm_group,
            )

            data.obs.append(ob)
            last_session = sname

    def _create_shared_fields(self, ob):
        ob.shared.create_column(self.times, shape=(ob.n_local_samples,), dtype=np.float64)
        ob.shared.create_column(
            self.position, shape=(ob.n_local_samples, 3), dtype=np.float64
        )
        ob.shared.create_column(
            self.velocity, shape=(ob.n_local_samples, 3), dtype=np.float64
        )
        ob.shared.create_column(
            self.azimuth, shape=(ob.n_local_samples,), dtype=np.float64
        )
        ob.shared.create_column(
            self.elevation, shape=(ob.n_local_samples,), dtype=np.float64
        )
        ob.shared.create_column(
            self.boresight_azel, shape=(ob.n_local_samples, 4), dtype=np.float64
        )
        ob.shared.create_column(
            self.boresight_radec, shape=(ob.n_local_samples, 4), dtype=np.float64
        )
        if self.shared_flags is not None:
            ob.shared.create_column(
                self.shared_flags, shape=(ob.n_local_samples,), dtype=np.uint8
            )

    def _simulate_scanning(self, scan, n_samples, rate, comm, samp_ranks):
        """Generate one smooth Lissajous AZ/EL trajectory for a schedule scan."""
        log = Logger.get()

        start = scan.start.timestamp()
        dt = np.arange(n_samples, dtype=np.float64) / rate
        times = start + dt

        az_min_sched = scan.az_min.to_value(u.radian)
        az_max_sched = scan.az_max.to_value(u.radian)
        az_center = 0.5 * (az_min_sched + az_max_sched)
        el_center = scan.el.to_value(u.radian)

        if self.lissajous_az_amplitude is None:
            az_amp = 0.5 * np.abs(az_max_sched - az_min_sched)
        else:
            az_amp = self.lissajous_az_amplitude.to_value(u.radian)

        el_amp = self.lissajous_el_amplitude.to_value(u.radian)
        az_freq = self.lissajous_az_rate.to_value(u.Hz)
        el_freq = self.lissajous_el_rate.to_value(u.Hz)
        az_phase = self.lissajous_az_phase.to_value(u.radian)
        el_phase = self.lissajous_el_phase.to_value(u.radian)
        az_drift = self.lissajous_az_drift_rate.to_value(u.radian / u.second)
        el_drift = self.lissajous_el_drift_rate.to_value(u.radian / u.second)

        self._check_lissajous_mount_limits(
            az_amp, el_amp, az_freq, el_freq, az_drift, el_drift, comm, log
        )

        az = az_center + az_drift * dt + az_amp * np.sin(
            2.0 * np.pi * az_freq * dt + az_phase
        )
        el = el_center + el_drift * dt + el_amp * np.sin(
            2.0 * np.pi * el_freq * dt + el_phase
        )

        sample_sets = self._make_sample_sets(n_samples, samp_ranks)
        if self.distribute_time and samp_ranks > len(sample_sets):
            if comm.group_rank == 0:
                msg = (
                    f"Group {comm.group} with {comm.group_size} processes cannot "
                    f"distribute {len(sample_sets)} sample sets."
                )
                log.error(msg)
                raise RuntimeError(msg)

        scan_interval = (times[0], times[-1])
        return (
            times,
            az,
            el,
            sample_sets,
            np.min(az),
            np.max(az),
            np.min(el),
            np.max(el),
            scan_interval,
        )

    def _check_lissajous_mount_limits(
        self, az_amp, el_amp, az_freq, el_freq, az_drift, el_drift, comm, log
    ):
        if not self.check_mount_motion_limits:
            return

        az_vmax = np.abs(az_drift) + 2.0 * np.pi * az_freq * np.abs(az_amp)
        el_vmax = np.abs(el_drift) + 2.0 * np.pi * el_freq * np.abs(el_amp)
        az_amax = (2.0 * np.pi * az_freq) ** 2 * np.abs(az_amp)
        el_amax = (2.0 * np.pi * el_freq) ** 2 * np.abs(el_amp)

        az_rate_lim = self.max_mount_rate_az.to_value(u.radian / u.second)
        az_accel_lim = self.max_mount_accel_az.to_value(u.radian / u.second**2)
        el_rate_lim = None
        el_accel_lim = None
        if self.max_mount_rate_el is not None:
            el_rate_lim = self.max_mount_rate_el.to_value(u.radian / u.second)
        if self.max_mount_accel_el is not None:
            el_accel_lim = self.max_mount_accel_el.to_value(u.radian / u.second**2)

        problems = []
        if az_vmax > az_rate_lim:
            problems.append(
                f"AZ peak rate {az_vmax / np.pi * 180.0:0.3f} deg/s exceeds "
                f"max_mount_rate_az {az_rate_lim / np.pi * 180.0:0.3f} deg/s"
            )
        if az_amax > az_accel_lim:
            problems.append(
                f"AZ peak accel {az_amax / np.pi * 180.0:0.3f} deg/s^2 exceeds "
                f"max_mount_accel_az {az_accel_lim / np.pi * 180.0:0.3f} deg/s^2"
            )
        if el_rate_lim is not None and el_vmax > el_rate_lim:
            problems.append(
                f"EL peak rate {el_vmax / np.pi * 180.0:0.3f} deg/s exceeds "
                f"max_mount_rate_el {el_rate_lim / np.pi * 180.0:0.3f} deg/s"
            )
        if el_accel_lim is not None and el_amax > el_accel_lim:
            problems.append(
                f"EL peak accel {el_amax / np.pi * 180.0:0.3f} deg/s^2 exceeds "
                f"max_mount_accel_el {el_accel_lim / np.pi * 180.0:0.3f} deg/s^2"
            )

        if len(problems) > 0:
            msg = "Invalid Lissajous mount motion limits:\n" + "\n".join(problems)
            if comm.group_rank == 0:
                log.error(msg)
            raise RuntimeError(msg)

    def _make_sample_sets(self, n_samples, samp_ranks):
        if n_samples <= 0:
            return []

        if self.sample_set_size > 0:
            lengths = []
            remaining = n_samples
            while remaining > 0:
                step = min(self.sample_set_size, remaining)
                lengths.append(step)
                remaining -= step
            return [[x] for x in lengths]

        if self.distribute_time:
            nset = max(1, samp_ranks)
        else:
            nset = 1

        base = n_samples // nset
        rem = n_samples % nset
        lengths = [base + (1 if i < rem else 0) for i in range(nset)]
        return [[x] for x in lengths if x > 0]

    def _azel_to_radec(self, site, stamps, bore_azel):
        return azel_to_radec(
            site,
            stamps,
            bore_azel,
            use_ephem=self.use_ephem,
            use_qpoint=self.use_qpoint,
        )

    def _obs_telescopes(self, data, det_ranks, detectors):
        """Split the session telescope by focalplane key, matching SimGround style."""
        log = Logger.get()

        session_tele_name = self.telescope.name
        session_tele_uid = self.telescope.uid
        site = self.telescope.site
        session_fp = self.telescope.focalplane

        if self.session_split_key is None and detectors is None:
            if self.detset_key is None:
                detsets = None
            else:
                detsets = session_fp.detector_groups(self.detset_key)
            return {"ALL": (self.telescope, detsets)}

        fp_input = session_fp.detector_data
        if detectors is None:
            keep_rows = [True for _ in range(len(fp_input))]
        else:
            dcheck = set(detectors)
            keep_rows = [x in dcheck for x in fp_input["name"]]

        if self.session_split_key is None:
            session_keys = ["ALL"]
        else:
            session_keys = np.unique(fp_input[self.session_split_key])

        obs_tele = {}
        for key in session_keys:
            if self.session_split_key is None:
                fp_rows = keep_rows
            else:
                fp_rows = np.logical_and(fp_input[self.session_split_key] == key, keep_rows)

            fp_detdata = QTable(fp_input[fp_rows])
            fp = Focalplane(
                detector_data=fp_detdata,
                sample_rate=session_fp.sample_rate,
                field_of_view=session_fp.field_of_view,
                thinfp=session_fp.thinfp,
            )

            if detectors is None:
                pipedets = fp.detectors
            else:
                check_det = set(detectors)
                pipedets = [det for det in fp.detectors if det in check_det]

            if self.detset_key is None:
                detsets = None
            else:
                dsets = fp.detector_groups(self.detset_key)
                if detectors is None:
                    detsets = dsets
                else:
                    pipe_check = set(pipedets)
                    detsets = {
                        dkey: [det for det in dval if det in pipe_check]
                        for dkey, dval in dsets.items()
                    }

            if not self.distribute_time:
                n_detset = len(pipedets) if detsets is None else len(detsets)
                if det_ranks > n_detset:
                    if data.comm.group_rank == 0:
                        msg = (
                            f"Group {data.comm.group} with {data.comm.group_size} "
                            f"processes cannot distribute {n_detset} detector sets."
                        )
                        log.error(msg)
                        raise RuntimeError(msg)

            safe_key = str(key).replace(" ", "-")
            if safe_key == "ALL":
                tele_name = session_tele_name
            else:
                tele_name = f"{session_tele_name}_{safe_key}"

            obs_tele[safe_key] = (
                Telescope(
                    tele_name,
                    uid=session_tele_uid,
                    focalplane=fp,
                    site=site,
                ),
                detsets,
            )

        return obs_tele

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        return {}

    def _provides(self):
        prov = {
            "shared": [
                self.times,
                self.shared_flags,
                self.azimuth,
                self.elevation,
                self.boresight_azel,
                self.boresight_radec,
                self.position,
                self.velocity,
            ],
            "detdata": [],
            "intervals": [
                self.scanning_interval,
                self.turnaround_interval,
                self.sun_up_interval,
                self.sun_close_interval,
            ],
        }
        prov["shared"] = [x for x in prov["shared"] if x is not None]
        if self.det_data is not None:
            prov["detdata"].append(self.det_data)
        if self.det_flags is not None:
            prov["detdata"].append(self.det_flags)
        return prov
