import pytest
from wavebuoy_nrt.wavebuoy import WaveBuoy


# def test_WaveBuoy():
#     wv = WaveBuoy(buoy_type="sofar", buoys_metadata_file_name="buoys_metadata.csv")
#     assert True

def test_get_site_ids():
    wv = WaveBuoy(buoy_type="sofar", buoys_metadata_file_name="buoys_metadata.csv")
    # assert type(wv.site_ids) is list
    print(wv.site_ids[0])
    assert type(wv.site_ids[0]) is dict

