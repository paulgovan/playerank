from .abstract import Feature
from .wyscoutEventsDefinition import (INTERRUPTION, FOUL, OFFSIDE, SHOT,
                                       DUEL, OTHERS,
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
    produces per-entity (player or team) per-match counts for eight features:

    Positive (credit for dangerous attacking sequences):
      chain-shot-participant      chains ending in a shot the entity appeared in
      chain-goal-participant      chains ending in a goal the entity appeared in
      chain-final-action          last non-shot touch before a shot (key-pass analog)

    Harmful (debit for sequences that precede or produce conceding):
      chain-turnover-precedes-shot   in a chain whose turnover immediately preceded
                                     an opponent shot
      chain-turnover-precedes-goal   same, but the opponent scored
      chain-turnover-final-actor     last actor before the turnover that led to
                                     an opponent shot/goal
      chain-conceded-shot            had a defensive action during an opp. shot chain
      chain-conceded-goal            had a defensive action during an opp. goal chain

    Chain-breaking rules (a new chain starts when any of these hold):
      - match period changes
      - current event is INTERRUPTION (5), FOUL (2), or OFFSIDE (6)
      - teamId changes between consecutive events (conservative)
    """

    # Tags that indicate possession loss on the preceding event
    LOSS_TAGS = frozenset([NOT_ACCURATE_TAG, DANGEROUS_BALL_LOST_TAG, INTERCEPTION_TAG])
    # Event IDs that unconditionally break a chain
    BREAK_EVENTS = frozenset([INTERRUPTION, FOUL, OFFSIDE])
    # Event IDs considered defensive actions for Type-2 (conceded chain) attribution
    DEFENSIVE_EVENTS = frozenset([DUEL, OTHERS])

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
            entity_features = self._compute_entity_features(
                chains, events, goalkeeper_ids, entity
            )

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

    def _get_entity(self, evt, goalkeeper_ids, entity):
        """Return the aggregation key for a single event, or None to skip."""
        if entity == 'player':
            pid = evt.get('playerId')
            return pid if pid and pid not in goalkeeper_ids else None
        return evt.get('teamId')

    def _compute_entity_features(self, chains, all_match_events, goalkeeper_ids, entity):
        """
        Accumulate positive and harmful chain counts per entity.

        Parameters
        ----------
        chains : list of list
            Possession chains for one match (output of _detect_chains).
        all_match_events : list
            All sorted events for the same match (needed for Type-2 attribution).
        goalkeeper_ids : frozenset
            Player IDs to exclude from player-level attribution.
        entity : str
            'player' or 'team'.
        """
        entity_features = defaultdict(lambda: defaultdict(int))

        # Pre-index defensive events by team for fast Type-2 window lookup
        def_events_by_team = defaultdict(list)
        for e in all_match_events:
            if e['eventId'] in self.DEFENSIVE_EVENTS:
                tid = e.get('teamId')
                if tid:
                    def_events_by_team[tid].append(e)

        teams_in_match = list({e['teamId'] for e in all_match_events
                                if e.get('teamId')})

        for i, chain in enumerate(chains):
            outcome = self._classify_chain(chain)
            team_id = chain[0]['teamId']

            # ── Positive features ─────────────────────────────────────────────
            if outcome in ('shot', 'goal'):
                # Collect unique entities in this chain (each credited once)
                participants = set()
                for evt in chain:
                    ent = self._get_entity(evt, goalkeeper_ids, entity)
                    if ent is not None:
                        participants.add(ent)

                for ent in participants:
                    entity_features[ent]['chain-shot-participant'] += 1
                    if outcome == 'goal':
                        entity_features[ent]['chain-goal-participant'] += 1

                # Last non-shot actor → final-action credit
                for evt in reversed(chain):
                    if evt['eventId'] == SHOT:
                        continue
                    ent = self._get_entity(evt, goalkeeper_ids, entity)
                    if ent is not None:
                        entity_features[ent]['chain-final-action'] += 1
                        break

            # ── Harmful Type 1: turnover chain ───────────────────────────────
            # This chain had no shot; the very next chain (different team) ended
            # in a shot or goal — our turnover gifted them the opportunity.
            if outcome is None and i + 1 < len(chains):
                next_chain = chains[i + 1]
                next_team = next_chain[0]['teamId']
                if next_team != team_id:
                    next_outcome = self._classify_chain(next_chain)
                    if next_outcome in ('shot', 'goal'):
                        suffix = ('precedes-goal' if next_outcome == 'goal'
                                  else 'precedes-shot')
                        feat = 'chain-turnover-' + suffix

                        participants = set()
                        for evt in chain:
                            ent = self._get_entity(evt, goalkeeper_ids, entity)
                            if ent is not None:
                                participants.add(ent)
                        for ent in participants:
                            entity_features[ent][feat] += 1

                        # Last actor in the turnover chain
                        for evt in reversed(chain):
                            ent = self._get_entity(evt, goalkeeper_ids, entity)
                            if ent is not None:
                                entity_features[ent]['chain-turnover-final-actor'] += 1
                                break

            # ── Harmful Type 2: conceded chain ───────────────────────────────
            # team_id's chain ended in shot/goal; debit the defending team's
            # players who had defensive actions in the chain's time window.
            if outcome in ('shot', 'goal'):
                defending_teams = [t for t in teams_in_match if t != team_id]
                if not defending_teams:
                    continue
                defending_team = defending_teams[0]
                feat = ('chain-conceded-goal' if outcome == 'goal'
                        else 'chain-conceded-shot')

                if entity == 'team':
                    entity_features[defending_team][feat] += 1
                else:
                    period_a = chain[0]['matchPeriod']
                    t_start  = chain[0]['eventSec']
                    t_end    = chain[-1]['eventSec']

                    defenders = {
                        e['playerId']
                        for e in def_events_by_team[defending_team]
                        if (e['matchPeriod'] == period_a
                            and t_start <= e['eventSec'] <= t_end
                            and e.get('playerId') not in goalkeeper_ids)
                    }
                    if defenders:
                        for pid in defenders:
                            entity_features[pid][feat] += 1
                    else:
                        # No specific defender identifiable; attribute to team
                        entity_features[defending_team][feat] += 1

        return entity_features
