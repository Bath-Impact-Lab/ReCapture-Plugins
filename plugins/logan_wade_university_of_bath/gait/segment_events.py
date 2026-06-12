# -*- coding: utf-8 -*-
"""Shared movement-agnostic segment event output helpers."""

import numpy as np


def _as_array_or_none(value):
    if value is None:
        return None
    return np.asarray(value)


def _event_set(segment_idx, segment_time, event_names, event_side):
    return {
        'Segment_Idx': _as_array_or_none(segment_idx),
        'Segment_Time': _as_array_or_none(segment_time),
        'Event_Names': list(event_names) if event_names is not None else [],
        'Event_Side': event_side
    }


def build_bilateral_segment_events(events, task, segment_label,
                                   idx_key='eventIdx', time_key='eventTime',
                                   names_key='eventNames'):
    return {
        'schema_version': 1,
        'task': task,
        'segment_label': segment_label,
        'default_event_set': 'bilateral',
        'event_sets': {
            'bilateral': _event_set(
                events[idx_key],
                events.get(time_key),
                events.get(names_key, []),
                'bilateral'
            )
        }
    }


def build_cycle_segment_events(events, task, segment_label):
    ipsilateral_side = events.get('ipsilateralLeg', 'r').lower()
    contralateral_side = 'l' if ipsilateral_side == 'r' else 'r'

    event_sets = {
        ipsilateral_side: _event_set(
            events['ipsilateralIdx'],
            events.get('ipsilateralTime'),
            events.get('eventNamesIpsilateral', []),
            ipsilateral_side
        )
    }

    if events.get('contralateralIdx') is not None:
        contralateral_event_names = (
            events.get('eventNamesContralateral') or
            events.get('eventNamesIpsilateral', [])
        )
        event_sets[contralateral_side] = _event_set(
            events['contralateralIdx'],
            events.get('contralateralTime'),
            contralateral_event_names,
            contralateral_side
        )

    return {
        'schema_version': 1,
        'task': task,
        'segment_label': segment_label,
        'default_event_set': ipsilateral_side,
        'event_sets': event_sets
    }


def build_bilateral_segment_events_from_cycle(events, task, segment_label):
    return {
        'schema_version': 1,
        'task': task,
        'segment_label': segment_label,
        'default_event_set': 'bilateral',
        'event_sets': {
            'bilateral': _event_set(
                events['ipsilateralIdx'],
                events.get('ipsilateralTime'),
                events.get('eventNamesIpsilateral', []),
                'bilateral'
            )
        }
    }
