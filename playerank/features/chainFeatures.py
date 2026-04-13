from .abstract import Feature
from .wyscoutEventsDefinition import (INTERRUPTION, FOUL, OFFSIDE, SHOT,
                                       GOAL_TAG, NOT_ACCURATE_TAG,
                                       DANGEROUS_BALL_LOST_TAG, INTERCEPTION_TAG)
import json
import glob
from collections import defaultdict


class chainFeatures(Feature):
    """
    Possession chain context features.

    Groups consecutive same-team events within a match into possession chains,
    classifies each chain by its terminal event (goal / shot / none), then
    produces three features per entity (player or team) per match:

    - chain-shot-participant : number of chains ending in a shot (incl. goals)
                               in which the entity touched the ball
    - chain-goal-participant : number of chains ending in a goal in which
                               the entity touched the ball
    - chain-final-action     : number of times the entity made the last
                               non-shot action before a shot/goal

    Chain-breaking rules (a new chain starts when any of these hold):
    - match period changes
    - current event is INTERRUPTION (5), FOUL (2), or OFFSIDE (6)
    - teamId changes between consecutive events
    """

    # Tags that indicate possession loss on the preceding event
    LOSS_TAGS = frozenset([NOT_ACCURATE_TAG, DANGEROUS_BALL_LOST_TAG, INTERCEPTION_TAG])
    # Event IDs that unconditionally break a chain
    BREAK_EVENTS = frozenset([INTERRUPTION, FOUL, OFFSIDE])

    def createFeature(self, events_path, players_file, entity='team'):
        """
        Compute chain features.

        Parameters
        ----------
        events_path : str
            Glob pattern for Wyscout events JSON files.
        players_file : str
            Path to players JSON file.
        entity : str
            'player' or 'team' — determines the aggregation key.

        Returns
        -------
        list of dict
            Records in the format: {match, entity, feature, value}
        """
        players = json.load(open(players_file))
        goalkeeper_ids = frozenset(
            p['wyId'] for p in players if p['role']['name'] == 'Goalkeeper'
        )

        match_events = defaultdict(list)
        for file in glob.glob(events_path):
            data = json.load(open(file))
            for evt in data:
                if evt['matchPeriod'] in ('1H', '2H'):
                    match_events[evt['matchId']].append(evt)
            print("[chainFeatures] loaded %s events from %s" % (len(data), file))

        result = []
        period_order = {'1H': 0, '2H': 1}

        for match_id, events in match_events.items():
            events.sort(
                key=lambda e: (period_order.get(e['matchPeriod'], 99), e['eventSec'])
            )

            chains = self._detect_chains(events)
            entity_features = self._compute_entity_features(chains, goalkeeper_ids, entity)

            for ent_id, feats in entity_features.items():
                for feat_name, value in feats.items():
                    if value > 0:
                        result.append({
                            'match': match_id,
                            'entity': ent_id,
                            'feature': feat_name,
                            'value': value,
                        })

        print("[chainFeatures] chain features computed. %s records produced" % len(result))
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_chains(self, events):
        """Split chronologically sorted events into possession chains."""
        if not events:
            return []

        chains = []
        current = [events[0]]

        for i in range(1, len(events)):
            prev = events[i - 1]
            curr = events[i]

            chain_break = (
                curr['matchPeriod'] != prev['matchPeriod']
                or curr['eventId'] in self.BREAK_EVENTS
                or curr['teamId'] != prev['teamId']
            )

            if chain_break:
                chains.append(current)
                current = [curr]
            else:
                current.append(curr)

        chains.append(current)
        return chains

    def _classify_chain(self, chain):
        """Return 'goal', 'shot', or None based on the terminal event."""
        last = chain[-1]
        if last['eventId'] != SHOT:
            return None
        tag_ids = {t['id'] for t in last.get('tags', [])}
        return 'goal' if GOAL_TAG in tag_ids else 'shot'

    def _compute_entity_features(self, chains, goalkeeper_ids, entity):
        """
        Accumulate chain participation counts per entity across all chains.
        """
        entity_features = defaultdict(lambda: defaultdict(int))

        for chain in chains:
            outcome = self._classify_chain(chain)
            if outcome is None:
                continue  # only credit chains that reach a shot

            # --- collect participants ---
            participants = set()
            for evt in chain:
                if entity == 'player':
                    pid = evt.get('playerId')
                    if pid and pid not in goalkeeper_ids:
                        participants.add(pid)
                else:
                    tid = evt.get('teamId')
                    if tid:
                        participants.add(tid)

            # --- find the last non-shot actor ---
            final_actor = None
            for evt in reversed(chain):
                if evt['eventId'] == SHOT:
                    continue
                if entity == 'player':
                    pid = evt.get('playerId')
                    if pid and pid not in goalkeeper_ids:
                        final_actor = pid
                        break
                else:
                    tid = evt.get('teamId')
                    if tid:
                        final_actor = tid
                        break

            # --- credit features ---
            for ent in participants:
                entity_features[ent]['chain-shot-participant'] += 1
                if outcome == 'goal':
                    entity_features[ent]['chain-goal-participant'] += 1

            if final_actor is not None:
                entity_features[final_actor]['chain-final-action'] += 1

        return entity_features
