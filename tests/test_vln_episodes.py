"""Offline contract tests; no model, private data, network or robot needed."""
from copy import deepcopy
import json
from pathlib import Path
import runpy
import shutil

import pytest

from go2_perception.vln.episodes import file_sha256, load_annotations, load_episode, select_history

FIXTURE = Path(__file__).parent / 'fixtures/vln/episode_v1'


@pytest.fixture
def package(tmp_path):
    shutil.copytree(FIXTURE, tmp_path/'episode')
    return tmp_path/'episode'


def alter(root, fn):
    path=root/'episode.json'
    data=json.loads(path.read_text()); fn(data)
    path.write_text(json.dumps(data))
    return path


def test_fixture_and_annotations_load():
    ep=load_episode(FIXTURE/'episode.json')
    anns=load_annotations(FIXTURE/'annotations.json',ep)
    assert anns['reviewer']['type']=='synthetic'
    assert 'annotations' not in ep.data
    s=select_history(ep,'d0')
    assert [f['frame_id'] for f in s['frames']]==['f0','f1','f3']
    assert s['missing_frame_ids']==['f2']
    assert s['gaps'][0]['reason']=='declared synthetic missing frame'


def test_fixture_is_deterministic(tmp_path):
    generator=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/generate_vln_episode_fixture.py'))
    generator['generate'](tmp_path/'fresh')
    for p in FIXTURE.rglob('*'):
        if p.is_file(): assert p.read_bytes()==(tmp_path/'fresh'/p.relative_to(FIXTURE)).read_bytes()


def test_availability_not_only_exposure_controls_causality(package):
    path=alter(package,lambda d:d['decisions'][0].update(time_s=103.05))
    s=select_history(load_episode(path),'d0')
    assert [f['frame_id'] for f in s['frames']]==['f0','f1']
    assert all(f['available_s']<=103.05 for f in s['frames'])


def test_current_only_and_uniform_history():
    ep=load_episode(FIXTURE/'episode.json')
    assert [f['frame_id'] for f in select_history(ep,'d1',max_frames=1)['frames']]==['f5']
    assert [f['frame_id'] for f in select_history(ep,'d1',max_frames=3)['frames']]==['f0','f3','f5']
    assert len(select_history(ep,'d0',max_frames=8)['frames'])==3
    with pytest.raises(ValueError):select_history(ep,'absent')


@pytest.mark.parametrize('mutation',[
    lambda d:d['frames'][1].update(clock_epoch='other_boot'),
    lambda d:d['frames'][1].update(recording_id='other_recording'),
    lambda d:d['frames'].reverse(),
    lambda d:d['frames'][1].update(observation_s=d['frames'][0]['observation_s']),
    lambda d:d['frames'][0].update(available_s=99.),
    lambda d:d['frames'][0].update(observation_s=99.),
    lambda d:d['decisions'][0].update(time_s=99.),
    lambda d:d['frames'][0].update(frame_id='f1'),
    lambda d:d.update(acceptable_actions=['STOP']),
    lambda d:d['frames'][0].update(target_visibility='visible'),
    lambda d:d['frames'][0].update(observation_s=float('nan')),
    lambda d:d['gaps'][0].update(end_s=200.),
])
def test_invalid_manifest_is_rejected(package,mutation):
    with pytest.raises(ValueError):load_episode(alter(package,mutation))


def test_declared_missing_vs_undeclared_file_loss(package):
    assert load_episode(package/'episode.json').data['frames'][2]['image'] is None
    (package/'images/000.png').unlink()
    with pytest.raises(ValueError,match='Missing or changed'):load_episode(package/'episode.json')


def test_hash_corruption_and_path_escape_rejected(package):
    (package/'images/000.png').write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):load_episode(package/'episode.json')
    path=alter(package,lambda d:d['frames'][0]['image'].update(path='../outside.png'))
    with pytest.raises(ValueError,match='relative'):load_episode(path,verify_files=False)


def test_annotation_hash_and_decision_coverage(package):
    ep=load_episode(package/'episode.json')
    path=package/'annotations.json';a=json.loads(path.read_text())
    a['manifest_sha256']='0'*64;path.write_text(json.dumps(a))
    with pytest.raises(ValueError,match='hash mismatch'):load_annotations(path,ep)
    a['manifest_sha256']=file_sha256(ep.path);a['decisions']=a['decisions'][:1];path.write_text(json.dumps(a))
    with pytest.raises(ValueError,match='each episode decision'):load_annotations(path,ep)


def test_future_gap_extent_not_leaked(package):
    path=alter(package,lambda d:d['gaps'][0].update(end_s=105.5))
    assert select_history(load_episode(path),'d0')['gaps'][0]['end_s']==103.2
