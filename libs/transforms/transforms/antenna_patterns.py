# from __future__ import annotations

# import argparse
# import csv
from collections.abc import Mapping
# from datetime import date
# import hashlib
# import json
import math
# from pathlib import Path
import re
# import warnings

import astropy
import astropy_iers_data
from astropy.utils import iers
from astropy.time import Time




DEFAULT_SKY_WINDOW_DEG = 1.0
CONVENTION = 'gwak_ra_minus_gmst_iau2006'


def _number(value, name):
    if isinstance(value, bool):
        raise ValueError(f'{name} must be a finite number, not a boolean')
    try:
        result = float(value)
    except (ValueError, TypeError) as e:
        raise ValueError(f'{name} must be a finite number') from e
    if not math.isfinite(result):
        raise ValueError(f'{name} must be finite')
    return result

def _utc_time(trigger_time, time_format):

    if time_format == 'isot':
        if not isinstance(trigger_time, str) or not re.fullmatch(
            r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?', trigger_time
        ):
            raise ValueError('Use a full UTC date/time: YYYY-MM-DDTHH:MM:SS[.fraction][Z]')
        value = trigger_time.rstrip('Z').replace(' ', 'T')
    elif time_format == 'mjd':
        value = _number(trigger_time, 'trigger_time (MJD UTC)')
    else:
        raise ValueError("time_format must be 'isot' or 'mjd'; both use UTC")
    return Time(value, format=time_format, scale='utc', precision=9)



def validate_sky_window(
    phi_rad,
    declination_rad,
    sky_window_deg=1.0 # DEFAULT_SKY_WINDOW_DEG
):
    """Validate a full-width phi/declination box; never silently clip at poles."""
    width = _number(sky_window_deg, 'sky_window_deg')
    if not 0 < width <= 360:
        raise ValueError('sky_window_deg must be in (0, 360] (full box width)')
    if (phi_rad is None) != (declination_rad is None):
        raise ValueError('fixed_phi_rad and fixed_declination_rad must be supplied together')
    if phi_rad is None:
        return None, None, width
    phi = _number(phi_rad, 'fixed_phi_rad')
    dec = _number(declination_rad, 'fixed_declination_rad')
    if not 0 <= phi < 2 * math.pi:
        raise ValueError('fixed_phi_rad must be in [0, 2*pi) radians')
    if abs(dec) > math.pi / 2:
        raise ValueError('fixed_declination_rad must be in [-pi/2, pi/2] radians')
    if abs(dec) + math.radians(width) / 2 > math.pi / 2:
        raise ValueError('Sky box crosses a pole; reduce sky_window_deg')
    return phi, dec, width


def event_to_gwak(
    ra_deg, dec_deg, 
    trigger_time, *, time_format='isot',
    sky_window_deg=DEFAULT_SKY_WINDOW_DEG, 
    temp_plot=False
):
    """Convert one event to JSON-compatible angles and ready-to-use sampler kwargs.
    ra_deg / dec_deg: numeric catalog equatorial coordinates in DEGREES.
    trigger_time: full UTC string, or numeric MJD with time_format='mjd'.
    sky_window_deg: full width in each coordinate, default 1 (= +/-0.5 deg).
    Uses Astropy mean Greenwich sidereal time (IAU2006) and bundled IERS data,
    without network downloads. Unsupported epochs fail rather than extrapolate.
    Conversion happens once; background timestamps do not affect sampling.
    temp_plot=True opens an unsaved window; close it to return the same result.
    Requires a local interactive Matplotlib backend. Default False does no plotting.
    """

    ra = _number(ra_deg, 'ra_deg')
    dec = _number(dec_deg, 'dec_deg')
    if not 0 <= ra < 360:
        raise ValueError('ra_deg must be in [0, 360) degrees')
    if not -90 <= dec <= 90:
        raise ValueError('dec_deg must be in [-90, 90] degrees')
    epoch = _utc_time(trigger_time, time_format)
    with iers.conf.set_temp('auto_download', False):
        dut1, status = epoch.get_delta_ut1_utc(return_status=True)
        if int(status) < 0:
            raise ValueError('Trigger time is outside bundled IERS coverage; update astropy-iers-data')
        # Record and use the identical UT1 correction, including predicted status.
        epoch.delta_ut1_utc = dut1.to_value('s')
        gmst = float(epoch.sidereal_time('mean', 'greenwich', model='IAU2006').rad)
    phi = (math.radians(ra) - gmst) % (2 * math.pi)
    phi, declination, width = validate_sky_window(
        phi,math.radians(dec),
        sky_window_deg
    )
    polar = math.pi / 2 - declination
    vector = [math.cos(declination)*math.cos(phi),
              math.cos(declination)*math.sin(phi), math.sin(declination)]
    result = {
        'input': {
            'ra_deg': ra,
            'dec_deg': dec,
            'trigger_time_utc': epoch.utc.isot,
            'mjd_utc': float(epoch.utc.mjd)
        },
        'convention': CONVENTION,
        'earth_phi_rad': phi,
        'earth_declination_rad': declination,
        'earth_polar_rad': polar,
        'earth_phi_deg': math.degrees(phi),
        'earth_declination_deg': dec,
        'earth_polar_deg': math.degrees(polar),
        'source_unit_vector': vector,
        'gmst_rad': gmst, 
        'ut1_minus_utc_seconds': float(dut1.to_value('s')),
        'iers_status': int(status), 
        'astropy_version': astropy.__version__,
        'astropy_iers_data_version': astropy_iers_data.__version__,
        'window': {
            'shape': 'phi_declination_box',
            'full_width_deg': width,
            'half_width_deg': width / 2
        },
        'gwak_sampling': {
            'fixed_phi_rad': phi,
            'fixed_declination_rad': declination,
            'sky_window_deg': width
        },
    }
    # if temp_plot:
    #     _show_temporary_figure(_position_figure(result, name="Event"))
    return result



def resolve_sky_position(
    *, 
    fixed_phi_rad=None,
    fixed_declination_rad=None,
    event_ra_deg=None,
    event_dec_deg=None,
    event_time_utc=None,
    sky_window_deg=DEFAULT_SKY_WINDOW_DEG,
    fixed_ra=None,
    fixed_dec=None
):
    """Resolve exactly one input mode: event, Earth-relative, legacy, or all-sky."""
    legacy = fixed_ra is not None or fixed_dec is not None
    earth = fixed_phi_rad is not None or fixed_declination_rad is not None

    # Todo: Look out for bette logic setup for _event and _condition
    # Can we implement a better guard clause
    # ToDo starts
    _event = any(
        v is not None 
        for v in (event_ra_deg, event_dec_deg, event_time_utc)
    )
    _condition = any(
        v is None 
        for v in (event_ra_deg, event_dec_deg, event_time_utc)
    )
    if sum((legacy, earth, _event)) > 1:
        raise ValueError(
            "Choose only one sky input mode: event, "
            "fixed Earth angles, or legacy aliases"
        )
    if _event:
        if _condition:
            raise ValueError(
                'event_ra_deg, event_dec_deg and event_time_utc are all required'
            )
    # ToDo ends
        return event_to_gwak(
            event_ra_deg,
            event_dec_deg,
            event_time_utc,
            sky_window_deg=sky_window_deg
        )

    if legacy:
        warnings.warn(
            'fixed_ra/fixed_dec mean Earth-relative radians, not celestial RA/Dec. '
            'Use fixed_phi_rad/fixed_declination_rad or event_* inputs.',
            DeprecationWarning, stacklevel=2
        )
        fixed_phi_rad, fixed_declination_rad = fixed_ra, fixed_dec
    phi, dec, width = validate_sky_window(
        fixed_phi_rad,
        fixed_declination_rad,
        sky_window_deg
    )
    return {
        'convention': CONVENTION if phi is not None else 'uniform_sky',
        'gwak_sampling': {
            'fixed_phi_rad': phi,
            'fixed_declination_rad': dec,
            'sky_window_deg': width
            }
        }

class SkyPositionMixin:
    """Small reusable sampler interface; torch is needed only when sampling."""
    def configure_sky_position(self, **kwargs):
        self.sky_position = resolve_sky_position(**kwargs)
        for name, value in self.sky_position['gwak_sampling'].items():
            setattr(self, name, value)

    @property
    def fixed_ra(self):
        """Deprecated mutable alias retained for archived evaluation scripts."""
        return self.fixed_phi_rad

    @fixed_ra.setter
    def fixed_ra(self, value):
        self.fixed_phi_rad = value
        self.sky_position = None  # explicit mutation invalidates conversion provenance

    @property
    def fixed_dec(self):
        return self.fixed_declination_rad

    @fixed_dec.setter
    def fixed_dec(self, value):
        self.fixed_declination_rad = value
        self.sky_position = None

    def _sample_sky_location(self, batch_size):
        import torch
        phi, dec, width = validate_sky_window(
            self.fixed_phi_rad, self.fixed_declination_rad, self.sky_window_deg)
        if phi is None:
            return self.ra_prior.sample((batch_size,)), self.dec_prior.sample((batch_size,))
        window = math.radians(width)
        # Keep two separate RNG draws in this order for legacy reproducibility.
        phis = (phi + (torch.rand(batch_size) - 0.5)*window) % (2*torch.pi)
        declinations = dec + (torch.rand(batch_size) - 0.5)*window
        return phis, declinations
