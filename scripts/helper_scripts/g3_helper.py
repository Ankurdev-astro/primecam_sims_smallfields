#Modified export_obs_meta from 
#https://github.com/hpc4cmb/toast/blob/e248f05a80428709fa006d69e793ae4fc5f8a557/src/toast/spt3g/spt3g_export.py#L189

import io
import os

import h5py
import numpy as np
from astropy import units as u

from toast.instrument import GroundSite, SpaceSite
from toast.intervals import IntervalList
from toast.timing import function_timer
from toast.utils import Environment, Logger, object_fullname
from toast.spt3g.spt3g_utils import (
    available,
    compress_timestream,
    to_g3_array_type,
    to_g3_map_array_type,
    to_g3_quats,
    to_g3_scalar_type,
    to_g3_time,
    to_g3_unit,
)

from spt3g import core as c3g

class export_obs_meta_mod(object):
    """Default class to export Observation and Calibration frames.

    This provides a default exporter of TOAST Observation metadata into Observation
    and Calibration frames.

    The telescope and site information will be written to the Observation frame.  The
    focalplane information will be written to the Calibration frame.

    The observation metadata is exported according to the following rules.
    Scalar values are converted to the closest appropriate G3 type and stored in the
    Observation frame.  Additional metadata arrays can be exported by specifying
    a `meta_arrays` list of tuples containing the TOAST observation name and the
    corresponding output Observation frame key name.  The arrays will be converted
    to the most appropriate type.

    Noise models can be exported to the Calibration frame by specifying a list of
    tuples containing the TOAST and frame keys.

    Args:
        meta_arrays (list):  The observation metadata arrays to export.
        noise_models (list):  The noise models to export.

    """

    def __init__(
        self,
        meta_arrays=list(),
        noise_models=list(),
    ):
        self._meta_arrays = meta_arrays
        self._noise_models = noise_models

    @function_timer
    def __call__(self, obs):
        log = Logger.get()
        log.verbose(f"Create observation frame for {obs.name}")
        # Construct observation frame
        ob = c3g.G3Frame(c3g.G3FrameType.Observation)
        ob["observation_name"] = c3g.G3String(obs.name)
        ob["observation_uid"] = c3g.G3Int(obs.uid)
        # ob["observation_detector_sets"] = c3g.G3VectorVectorString(
        #     obs.all_detectors
        # )
        ob["observation_detector_sets"] = c3g.G3VectorVectorString(
            [[det] for det in obs.all_detectors]
        )
        ob["telescope_name"] = c3g.G3String(obs.telescope.name)
        ob["telescope_class"] = c3g.G3String(object_fullname(obs.telescope.__class__))
        ob["telescope_uid"] = c3g.G3Int(obs.telescope.uid)
        site = obs.telescope.site
        ob["site_name"] = c3g.G3String(site.name)
        ob["site_class"] = c3g.G3String(object_fullname(site.__class__))
        ob["site_uid"] = c3g.G3Int(site.uid)
        if isinstance(site, GroundSite):
            ob["site_lat_deg"] = c3g.G3Double(site.earthloc.lat.to_value(u.degree))
            ob["site_lon_deg"] = c3g.G3Double(site.earthloc.lon.to_value(u.degree))
            ob["site_alt_m"] = c3g.G3Double(site.earthloc.height.to_value(u.meter))
            if site.weather is not None:
                if hasattr(site.weather, "name"):
                    # This is a simulated weather object, dump it.
                    ob["site_weather_name"] = c3g.G3String(site.weather.name)
                    ob["site_weather_realization"] = c3g.G3Int(site.weather.realization)
                    if site.weather.max_pwv is None:
                        ob["site_weather_max_pwv"] = c3g.G3String("NONE")
                    else:
                        ob["site_weather_max_pwv"] = c3g.G3Double(site.weather.max_pwv.to_value(u.mm))
                    ob["site_weather_time"] = to_g3_time(site.weather.time.timestamp())
                    ob["site_weather_uid"] = c3g.G3Int(site.weather.site_uid)
                    ob["site_weather_use_median"] = c3g.G3Bool(
                        site.weather.median_weather
                    )
        session = obs.session
        if session is not None:
            ob["session_name"] = c3g.G3String(session.name)
            ob["session_class"] = c3g.G3String(object_fullname(session.__class__))
            ob["session_uid"] = c3g.G3Int(session.uid)
            if session.start is None:
                ob["session_start"] = c3g.G3String("NONE")
            else:
                ob["session_start"] = to_g3_time(session.start.timestamp())
            if session.end is None:
                ob["session_end"] = c3g.G3String("NONE")
            else:
                ob["session_end"] = to_g3_time(session.end.timestamp())
        m_export = set()
        for m_in, m_out in self._meta_arrays:
            out_type = to_g3_array_type(obs[m_in].dtype)
            ob[m_out] = out_type(obs[m_in])
            m_export.add(m_in)
        for m_key, m_val in obs.items():
            if m_key in m_export:
                # already done
                continue
            try:
                l = len(m_val)
                # This is an array
            except Exception:
                # This is a scalar (no len defined)
                try:
                    ob[m_key], m_unit = to_g3_scalar_type(m_val)
                    if m_unit is not None:
                        ob[f"{m_key}_astropy_units"] = c3g.G3String(f"{m_val.unit}")
                        ob[f"{m_key}_units"] = m_unit
                except Exception:
                    # This is not a datatype we can convert
                    pass

        # Construct calibration frame
        cal = c3g.G3Frame(c3g.G3FrameType.Calibration)

        # Serialize focalplane to HDF5 bytes and write to frame.
        byte_writer = io.BytesIO()
        with h5py.File(byte_writer, "w") as f:
            obs.telescope.focalplane.save_hdf5(f, comm=None, force_serial=True)
        cal["focalplane"] = c3g.G3VectorUnsignedChar(byte_writer.getvalue())
        del byte_writer

        # Serialize noise models
        for m_in, m_out in self._noise_models:
            byte_writer = io.BytesIO()
            with h5py.File(byte_writer, "w") as f:
                obs[m_in].save_hdf5(f, obs)
            cal[m_out] = c3g.G3VectorUnsignedChar(byte_writer.getvalue())
            del byte_writer
            cal[f"{m_out}_class"] = c3g.G3String(object_fullname(obs[m_in].__class__))

        return ob, cal
