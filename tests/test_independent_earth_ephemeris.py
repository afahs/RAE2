import numpy as np
import pandas as pd
import pytest
from astropy import units as u
from astropy.coordinates import FK4, SkyCoord
from astropy.time import Time

from rylevonberg.constants import EARTH_UNIT_COLUMNS, SPACECRAFT_COLUMNS, GEOMETRY_VERSION
from rylevonberg.events import source_vectors_for_rows, predict_events
from rylevonberg.frames import body_positions_from_moon, body_vectors_from_spacecraft, body_unit_vectors_from_spacecraft
from rylevonberg.geometry import normalize_vectors, moon_limb_angle_deg
from rylevonberg.ingest import IngestOptions, ingest_csv, validate_timeseries


def sample():
    t=pd.date_range('1974-02-13 08:00',periods=4,freq='20min')
    p=np.array([[2800.,0,0],[0,2800.,0],[-2800.,0,0],[0,-2800.,0]])
    return t,p,pd.DataFrame({'time':t,**dict(zip(SPACECRAFT_COLUMNS,p.T))})


def test_earth_directions_ignore_missing_or_corrupt_archived_field():
    t,p,df=sample();source=pd.Series({'source_name':'earth','kind':'earth'})
    expected=source_vectors_for_rows(source,t,df)
    for values in (np.full((4,3),np.nan),np.zeros((4,3)),np.tile([1.,0,0],(4,1)),np.ones((4,3))*1e30):
        bad=df.assign(**dict(zip(EARTH_UNIT_COLUMNS,values.T)))
        np.testing.assert_array_equal(source_vectors_for_rows(source,t,bad),expected)
    assert np.isfinite(expected).all()


def test_observer_parallax_and_actual_range_are_preserved():
    t,p,_=sample();moon=body_positions_from_moon('earth',t)
    sc=body_vectors_from_spacecraft('earth',t,p)
    np.testing.assert_allclose(sc+p,moon,atol=1e-9)
    distance=np.linalg.norm(moon,axis=1)
    assert np.all((distance>350000)&(distance<420000))
    assert np.ptp(distance)>1.
    assert np.max(np.linalg.norm(normalize_vectors(sc)-normalize_vectors(moon),axis=1))>.003


@pytest.mark.parametrize('frame,equinox',[('fk4','B1950'),('fk5','J2000')])
def test_transform_changes_axes_not_body_origin_or_range(frame,equinox):
    t,_,_=sample();icrs=body_positions_from_moon('earth',t,'icrs');target=body_positions_from_moon('earth',t,frame,equinox)
    from rylevonberg.frames import frame_object
    axes=FK4(equinox=Time(equinox),obstime=Time(t)) if frame=='fk4' else frame_object(frame,equinox)
    c=SkyCoord(x=target[:,0]*u.km,y=target[:,1]*u.km,z=target[:,2]*u.km,representation_type='cartesian',frame=axes)
    np.testing.assert_allclose(c.icrs.cartesian.xyz.to_value(u.km).T,icrs,atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(target,axis=1),np.linalg.norm(icrs,axis=1),rtol=1e-9)


def test_time_units_duplicates_timezone_and_missing_rows():
    t,_,_=sample();ref=body_positions_from_moon('earth',t)
    for unit in ('s','ms','us','ns'):
        np.testing.assert_allclose(body_positions_from_moon('earth',t.as_unit(unit)),ref,atol=1e-8)
    mixed=pd.DatetimeIndex([t[2],pd.NaT,t[0],t[2]])
    out=body_positions_from_moon('earth',mixed)
    assert np.isnan(out[1]).all();np.testing.assert_array_equal(out[0],out[3]);np.testing.assert_array_equal(out[[0,2]],ref[[2,0]])
    np.testing.assert_array_equal(body_positions_from_moon('earth',t.tz_localize('UTC').tz_convert('US/Pacific')),ref)
    assert body_positions_from_moon('earth',pd.DatetimeIndex([])).shape==(0,3)


def test_earth_alias_and_offsets_share_observer_geometry():
    t,p,df=sample();a=source_vectors_for_rows(pd.Series({'source_name':'earth','kind':'earth'}),t,df)
    b=source_vectors_for_rows(pd.Series({'source_name':'earth','kind':'body','body_name':'earth'}),t,df)
    c=source_vectors_for_rows(pd.Series({'source_name':'earth_control','kind':'body_offset','body_name':'earth','offset_east_deg':0,'offset_north_deg':0}),t,df)
    np.testing.assert_array_equal(a,b);np.testing.assert_allclose(a,c,atol=1e-15)


def test_event_veto_and_predictions_do_not_depend_on_archive_earth():
    t=pd.date_range('1974-02-13 08:00',periods=101,freq='15s')
    earth=normalize_vectors(body_positions_from_moon('earth',t));a=earth[0]
    cross=np.cross(a,[0,0,1.]);cross/=np.linalg.norm(cross)
    theta=np.arcsin(1737.4/2800)+np.linspace(-.1,.1,len(t))
    p=2800*(-np.cos(theta)[:,None]*a+np.sin(theta)[:,None]*cross)
    df=pd.DataFrame({'time':t,**dict(zip(SPACECRAFT_COLUMNS,p.T))})
    target=pd.DataFrame([{'source_name':'near_earth','kind':'fixed','frame':'fk4','ra_deg':np.degrees(np.arctan2(a[1],a[0]))%360,'dec_deg':np.degrees(np.arcsin(a[2]))}])
    contaminant=pd.DataFrame([{'source_name':'earth','kind':'earth'}])
    events,states=predict_events(df,target)
    assert len(events)==1 and states.geometry_version.eq(GEOMETRY_VERSION).all()
    for value in (np.nan,0.,1e20):
        modified=df.assign(**{c:value for c in EARTH_UNIT_COLUMNS})
        vetoed,_=predict_events(modified,target,limb_exclusion_sources_df=contaminant,limb_exclusion_deg=2.)
        assert vetoed.empty


def test_ingest_retains_power_without_archived_earth_columns(tmp_path):
    t,p,df=sample();df['frequency_band']=4;df['right_ascension']=1.;df['declination']=20.;df['rv1_fine']=[100.,110.,90.,105.]
    path=tmp_path/'native.csv';df.to_csv(path,index=False)
    clean,_=ingest_csv(path,IngestOptions(value_columns=('rv1_fine',),use_existing_loader=False))
    assert clean.is_valid.all()
    corrupted=clean.assign(**{c:np.nan for c in EARTH_UNIT_COLUMNS})
    checked,_=validate_timeseries(corrupted)
    assert checked.is_valid.all()
    assert checked.quality_flags.str.contains('archive_earth_invalid').all()


def test_old_stack_version_is_not_reused():
    from rylevonberg.stacking import stack_cache_matches
    old=pd.DataFrame({'stack_processing_version':['source_preserving_v3'],'baseline_mode':['joint_step_linear'],'window_s':[600.],'sideband_exclusion_seconds':[120.]})
    assert not stack_cache_matches(old,600.)


def test_event_cache_requires_current_geometry():
    from rylevonberg.events import event_cache_matches,require_current_geometry
    old=pd.DataFrame({'predicted_event_time':[pd.Timestamp('1974-01-01')]})
    assert not event_cache_matches(old)
    with pytest.raises(ValueError,match='rerun predict'):
        require_current_geometry(old)
    current=old.assign(geometry_version=GEOMETRY_VERSION)
    assert event_cache_matches(current)
    require_current_geometry(current)


def test_old_quality_flags_require_reingest_before_pipeline_analysis(tmp_path):
    from rylevonberg.ingest import QUALITY_POLICY_VERSION,require_current_quality
    from rylevonberg.pipeline import _read_current_cleaned
    old=pd.DataFrame({'time':[pd.Timestamp('1974-01-01')],
                      'quality_policy_version':['instrument_limits_v2']})
    path=tmp_path/'old.csv';old.to_csv(path,index=False)
    with pytest.raises(ValueError,match='rerun ingest'):
        _read_current_cleaned(path)
    require_current_quality(old.assign(quality_policy_version=QUALITY_POLICY_VERSION))
