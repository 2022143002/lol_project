import os
import re
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import json

def get_default_match_data(match, timeline, challengers):
    data = {}

    #게임의 기본정보
    #data는 우리가 원하는 정보를 추출하는데 필요한 틀이다.
    data["matchId"] = match["metadata"]["matchId"]
    data["gameDurationSec"] = match["info"]["gameDuration"]
    data["numFrames"] = len(timeline["info"]["frames"])
    tmp_teamId = match["info"]["participants"][0]["teamId"]
    tmp_win = match["info"]["participants"][0]["win"]
    data["blueWin"] = tmp_win if tmp_teamId == 100 else (not tmp_win)

    # 데이터 롤 초기화
    lanes = ["TOP", "MIDDLE", "JUNGLE", "BOTTOM"]
    for lane in lanes:
        data[lane] = {"blue": {"feature":{}, "target":{}}, "red":{"feature":{}, "target":{}}}  # [추가] target 칸
    data["BOTTOM"]["blue"]["participantId"] = {}
    data["BOTTOM"]["blue"]["challengerPuuid"] = {}
    data["BOTTOM"]["red"]["participantId"] = {}
    data["BOTTOM"]["red"]["challengerPuuid"] = {}

    # lane별로 feature를 제외한 기본정보를 입려해준다
    # 기본정보는 putricipantId, challengerPuuid, teamId(Blue / Red)
    # lane을 구분하기 위해서 match["info"]["participants"]를 참조해 보면
    # lane, indinisualPosition, teamPosition, role 등이 있는데
    # 서로 구분되는 역할을 명시한 키 값은 teamPosition 이다.
    for p in match["info"]["participants"]:
        team = p["teamId"]
        side = "blue" if team == 100 else "red"
        pos = p["teamPosition"]
        pid = p["participantId"]
        puid = p["puuid"]
        is_chall = puid in challengers
        chall = puid if is_chall else None

        if pos == "TOP":
            data["TOP"][side]["teamId"] = team
            data["TOP"][side]["participantId"] = pid
            data["TOP"][side]["challengerPuuid"] = chall
        elif pos == "MIDDLE":
            data["MIDDLE"][side]["teamId"] = team
            data["MIDDLE"][side]["participantId"] = pid
            data["MIDDLE"][side]["challengerPuuid"] = chall
        elif pos == "JUNGLE":
            data["JUNGLE"][side]["teamId"] = team
            data["JUNGLE"][side]["participantId"] = pid
            data["JUNGLE"][side]["challengerPuuid"] = chall
        elif pos == "BOTTOM":
            data["BOTTOM"][side]["teamId"] = team
            data["BOTTOM"][side]["participantId"]["ADC"] = pid
            data["BOTTOM"][side]["challengerPuuid"]["ADC"] = chall
        elif pos == "UTILITY":
            data["BOTTOM"][side]["teamId"] = team
            data["BOTTOM"][side]["participantId"]["SPT"] = pid
            data["BOTTOM"][side]["challengerPuuid"]["SPT"] = chall
        else:
            print(f"Something error: {pos}")
            pass

    return data

# =====================================================================
# [변경] 라인별 피처(x) 추출
# ---------------------------------------------------------------------
# data_process_sample.py 는 누적 / 격차 / 시점 / 이벤트 방식을 예시로 보여주고
# "세 가지를 섞지 말고 하나로 통일" 할 것을 제안한다.
# → 모든 피처를 타겟과 같은 '누적 격차' 방식으로 통일한다.
#     블루 : 블루 누적 - 레드 누적,  레드 : 레드 누적 - 블루 누적
#
# 선정 원칙
#   (1) 같은 라인의 타겟 지표는 피처로 쓰지 않는다 (자기 참조 방지)
#   (2) 타겟 지표의 '결과'를 담은 지표도 쓰지 않는다
#       예) JUNGLE/MIDDLE 의 골드·경험치에는 킬/어시스트 보상이 섞여 있으므로 제외
#   (3) 타겟을 '만들어내는 원인' 이 되는 플레이 지표는 사용한다
#       예) BOTTOM 의 킬관여는 골드를 버는 원인이므로 사용
#
#   TOP    : 미니언, 가한 피해, 받은 피해, 솔로킬, 데스
#   JUNGLE : 정글 몬스터, 에픽 몬스터, 데스, 와드 설치, 가한 피해
#   MIDDLE : 미니언, 가한 피해, 받은 피해, 데스, 방패 파괴, 로밍 시간
#   BOTTOM : ADC 가한 피해, 듀오 킬관여, 듀오 데스, 듀오 와드 제거, 방패 파괴
# =====================================================================

EPIC_MONSTERS = ["DRAGON", "RIFTHERALD", "HORDE", "BARON_NASHOR", "ATAKHAN"]


def diff_pair(blue_value, red_value, blue_list, red_list):
    """격차를 블루/레드 관점으로 각각 저장 (블루 : b - r, 레드 : r - b)"""
    blue_list.append(blue_value - red_value)
    red_list.append(red_value - blue_value)


def get_top_feature(match, timeline, data):
    # 1. 미니언 격차     : minionsKilled (누적)
    # 2. 가한 피해 격차  : damageStats.totalDamageDoneToChampions (누적)
    # 3. 받은 피해 격차  : damageStats.totalDamageTaken (누적)
    # 4. 솔로킬 격차     : CHAMPION_KILL 중 어시스트 없이 맞라이너를 처치 (이벤트 누적)
    # 5. 데스 격차       : CHAMPION_KILL 의 victimId (이벤트 누적)
    minionB, minionR, dmgDoneB, dmgDoneR, dmgTakenB, dmgTakenR = [], [], [], [], [], []
    soloKillB, soloKillR, deathB, deathR = [], [], [], []
    soloKillBlue, soloKillRed, deathBlue, deathRed = 0, 0, 0, 0

    bluePid = data["TOP"]["blue"]["participantId"]
    redPid = data["TOP"]["red"]["participantId"]

    for frame in timeline["info"]["frames"]:
        pf = frame["participantFrames"]
        b, r = pf[str(bluePid)], pf[str(redPid)]

        diff_pair(b["minionsKilled"], r["minionsKilled"], minionB, minionR)
        diff_pair(b["damageStats"]["totalDamageDoneToChampions"],
                  r["damageStats"]["totalDamageDoneToChampions"], dmgDoneB, dmgDoneR)
        diff_pair(b["damageStats"]["totalDamageTaken"],
                  r["damageStats"]["totalDamageTaken"], dmgTakenB, dmgTakenR)

        for event in frame["events"]:
            if event["type"] != "CHAMPION_KILL":
                continue
            if "assistingParticipantIds" not in event:
                if event["killerId"] == bluePid and event["victimId"] == redPid:
                    soloKillBlue += 1
                elif event["killerId"] == redPid and event["victimId"] == bluePid:
                    soloKillRed += 1
            if event["victimId"] == bluePid:
                deathBlue += 1
            elif event["victimId"] == redPid:
                deathRed += 1
        diff_pair(soloKillBlue, soloKillRed, soloKillB, soloKillR)
        diff_pair(deathBlue, deathRed, deathB, deathR)

    data["TOP"]["blue"]["feature"].update({
        "minionDiff": minionB, "dmgDoneDiff": dmgDoneB, "dmgTakenDiff": dmgTakenB,
        "soloKillDiff": soloKillB, "deathDiff": deathB})
    data["TOP"]["red"]["feature"].update({
        "minionDiff": minionR, "dmgDoneDiff": dmgDoneR, "dmgTakenDiff": dmgTakenR,
        "soloKillDiff": soloKillR, "deathDiff": deathR})


def get_jungle_feature(match, timeline, data):
    # 1. 정글 몬스터 격차 : jungleMinionsKilled (누적)
    # 2. 에픽 몬스터 격차 : ELITE_MONSTER_KILL (드래곤, 전령, 공허 유충, 바론, 아타칸) 팀 기준 누적
    #                       에픽 몬스터는 강타를 가진 정글러가 주도하므로 정글러의 성과로 본다.
    # 3. 데스 격차        : victimId (이벤트 누적)
    # 4. 와드 설치 격차   : WARD_PLACED 의 creatorId (UNDEFINED 제외, 이벤트 누적)
    # 5. 가한 피해 격차   : totalDamageDoneToChampions (누적)
    jungleB, jungleR, epicB, epicR, deathB, deathR = [], [], [], [], [], []
    wardB, wardR, dmgDoneB, dmgDoneR = [], [], [], []
    epicBlue, epicRed, deathBlue, deathRed, wardBlue, wardRed = 0, 0, 0, 0, 0, 0

    bluePid = data["JUNGLE"]["blue"]["participantId"]
    redPid = data["JUNGLE"]["red"]["participantId"]

    for frame in timeline["info"]["frames"]:
        pf = frame["participantFrames"]
        b, r = pf[str(bluePid)], pf[str(redPid)]

        diff_pair(b["jungleMinionsKilled"], r["jungleMinionsKilled"], jungleB, jungleR)
        diff_pair(b["damageStats"]["totalDamageDoneToChampions"],
                  r["damageStats"]["totalDamageDoneToChampions"], dmgDoneB, dmgDoneR)

        for event in frame["events"]:
            if event["type"] == "ELITE_MONSTER_KILL" and event.get("monsterType") in EPIC_MONSTERS:
                if event.get("killerTeamId") == 100:
                    epicBlue += 1
                elif event.get("killerTeamId") == 200:
                    epicRed += 1
            elif event["type"] == "CHAMPION_KILL":
                if event["victimId"] == bluePid:
                    deathBlue += 1
                elif event["victimId"] == redPid:
                    deathRed += 1
            elif event["type"] == "WARD_PLACED" and event.get("wardType") != "UNDEFINED":
                if event["creatorId"] == bluePid:
                    wardBlue += 1
                elif event["creatorId"] == redPid:
                    wardRed += 1
        diff_pair(epicBlue, epicRed, epicB, epicR)
        diff_pair(deathBlue, deathRed, deathB, deathR)
        diff_pair(wardBlue, wardRed, wardB, wardR)

    data["JUNGLE"]["blue"]["feature"].update({
        "jungleMinionDiff": jungleB, "epicMonsterDiff": epicB, "deathDiff": deathB,
        "wardPlacedDiff": wardB, "dmgDoneDiff": dmgDoneB})
    data["JUNGLE"]["red"]["feature"].update({
        "jungleMinionDiff": jungleR, "epicMonsterDiff": epicR, "deathDiff": deathR,
        "wardPlacedDiff": wardR, "dmgDoneDiff": dmgDoneR})


ROAM_DIST = 1500   # 미드 라인 중심선(x = y)에서 이 거리 이상 떨어져 있으면 로밍으로 판단


def is_roaming(frame_pf):
    """미드 라인은 블루 기지(좌하단)와 레드 기지(우상단)를 잇는 대각선 x = y 위에 있다.
    점 (x, y) 와 직선 x = y 사이의 거리는 |x - y| / √2 이다."""
    x, y = frame_pf["position"]["x"], frame_pf["position"]["y"]
    return abs(x - y) / 2 ** 0.5 > ROAM_DIST


def get_middle_feature(match, timeline, data):
    # 1. 미니언 격차    : minionsKilled (누적)
    # 2. 가한 피해 격차 : totalDamageDoneToChampions (누적)
    # 3. 받은 피해 격차 : totalDamageTaken (누적)
    # 4. 데스 격차      : victimId (이벤트 누적)
    # 5. 방패 파괴 격차 : TURRET_PLATE_DESTROYED 중 MID_LANE (팀 기준, 이벤트 누적)
    #                     teamId 는 '잃은 팀' 이므로 teamId == 200 이면 블루가 깬 것
    # 6. 로밍 시간 격차 : 매 분 위치(position)가 미드 라인에서 ROAM_DIST 이상 떨어져 있던 분 수 (누적)
    #                     위치는 1분에 한 번 기록되므로 '로밍 중이던 분의 수' 로 근사한다.
    minionB, minionR, dmgDoneB, dmgDoneR, dmgTakenB, dmgTakenR = [], [], [], [], [], []
    deathB, deathR, plateB, plateR, roamB, roamR = [], [], [], [], [], []
    deathBlue, deathRed, plateBlue, plateRed, roamBlue, roamRed = 0, 0, 0, 0, 0, 0

    bluePid = data["MIDDLE"]["blue"]["participantId"]
    redPid = data["MIDDLE"]["red"]["participantId"]

    for t, frame in enumerate(timeline["info"]["frames"]):
        pf = frame["participantFrames"]
        b, r = pf[str(bluePid)], pf[str(redPid)]

        diff_pair(b["minionsKilled"], r["minionsKilled"], minionB, minionR)
        diff_pair(b["damageStats"]["totalDamageDoneToChampions"],
                  r["damageStats"]["totalDamageDoneToChampions"], dmgDoneB, dmgDoneR)
        diff_pair(b["damageStats"]["totalDamageTaken"],
                  r["damageStats"]["totalDamageTaken"], dmgTakenB, dmgTakenR)

        if t > 0:   # 0분은 기지에서 출발하기 전이므로 제외
            roamBlue += is_roaming(b)
            roamRed += is_roaming(r)
        diff_pair(roamBlue, roamRed, roamB, roamR)

        for event in frame["events"]:
            if event["type"] == "CHAMPION_KILL":
                if event["victimId"] == bluePid:
                    deathBlue += 1
                elif event["victimId"] == redPid:
                    deathRed += 1
            elif event["type"] == "TURRET_PLATE_DESTROYED" and event.get("laneType") == "MID_LANE":
                if event["teamId"] == 200:
                    plateBlue += 1
                else:
                    plateRed += 1
        diff_pair(deathBlue, deathRed, deathB, deathR)
        diff_pair(plateBlue, plateRed, plateB, plateR)

    data["MIDDLE"]["blue"]["feature"].update({
        "minionDiff": minionB, "dmgDoneDiff": dmgDoneB, "dmgTakenDiff": dmgTakenB,
        "deathDiff": deathB, "plateDiff": plateB, "roamDiff": roamB})
    data["MIDDLE"]["red"]["feature"].update({
        "minionDiff": minionR, "dmgDoneDiff": dmgDoneR, "dmgTakenDiff": dmgTakenR,
        "deathDiff": deathR, "plateDiff": plateR, "roamDiff": roamR})


def get_bottom_feature(match, timeline, data):
    # ADC + SPT 듀오
    # 1. ADC 가한 피해 격차 : ADC 의 totalDamageDoneToChampions (누적)
    # 2. 듀오 킬관여 격차   : 듀오 중 한 명이라도 킬/어시스트로 참여한 처치 수 (이벤트 누적)
    # 3. 듀오 데스 격차     : 듀오의 데스 합 (이벤트 누적)
    # 4. 듀오 와드 제거 격차 : WARD_KILL 의 killerId (이벤트 누적)
    # 5. 방패 파괴 격차     : TURRET_PLATE_DESTROYED 중 BOT_LANE (팀 기준, 이벤트 누적)
    adcDmgB, adcDmgR, takedownB, takedownR, deathB, deathR = [], [], [], [], [], []
    wardKillB, wardKillR, plateB, plateR = [], [], [], []
    takedownBlue, takedownRed, deathBlue, deathRed = 0, 0, 0, 0
    wardKillBlue, wardKillRed, plateBlue, plateRed = 0, 0, 0, 0

    blueAdc = data["BOTTOM"]["blue"]["participantId"]["ADC"]
    redAdc = data["BOTTOM"]["red"]["participantId"]["ADC"]
    bluePids = list(data["BOTTOM"]["blue"]["participantId"].values())
    redPids = list(data["BOTTOM"]["red"]["participantId"].values())

    for frame in timeline["info"]["frames"]:
        pf = frame["participantFrames"]
        diff_pair(pf[str(blueAdc)]["damageStats"]["totalDamageDoneToChampions"],
                  pf[str(redAdc)]["damageStats"]["totalDamageDoneToChampions"], adcDmgB, adcDmgR)

        for event in frame["events"]:
            if event["type"] == "CHAMPION_KILL":
                involved = [event["killerId"]] + event.get("assistingParticipantIds", [])
                if any(p in involved for p in bluePids):
                    takedownBlue += 1
                if any(p in involved for p in redPids):
                    takedownRed += 1
                if event["victimId"] in bluePids:
                    deathBlue += 1
                elif event["victimId"] in redPids:
                    deathRed += 1
            elif event["type"] == "WARD_KILL":
                if event["killerId"] in bluePids:
                    wardKillBlue += 1
                elif event["killerId"] in redPids:
                    wardKillRed += 1
            elif event["type"] == "TURRET_PLATE_DESTROYED" and event.get("laneType") == "BOT_LANE":
                if event["teamId"] == 200:
                    plateBlue += 1
                else:
                    plateRed += 1
        diff_pair(takedownBlue, takedownRed, takedownB, takedownR)
        diff_pair(deathBlue, deathRed, deathB, deathR)
        diff_pair(wardKillBlue, wardKillRed, wardKillB, wardKillR)
        diff_pair(plateBlue, plateRed, plateB, plateR)

    data["BOTTOM"]["blue"]["feature"].update({
        "adcDmgDoneDiff": adcDmgB, "duoTakedownDiff": takedownB, "duoDeathDiff": deathB,
        "duoWardKillDiff": wardKillB, "plateDiff": plateB})
    data["BOTTOM"]["red"]["feature"].update({
        "adcDmgDoneDiff": adcDmgR, "duoTakedownDiff": takedownR, "duoDeathDiff": deathR,
        "duoWardKillDiff": wardKillR, "plateDiff": plateR})


# =====================================================================
# [추가] 라인별 타겟 변수(y) 추출
# ---------------------------------------------------------------------
# 타겟은 모두 '격차' 방식 (블루 : 블루 - 레드, 레드 : 레드 - 블루) 으로 통일하고,
# 위의 xpDiffCum 예시처럼 분 단위 리스트로 data[lane][side]["target"] 에 저장한다.
#   TOP    : 경험치 격차 + 포탑 골드 격차
#   JUNGLE : 킬 격차 + 어시스트 격차
#   MIDDLE : 킬 격차 + 어시스트 격차
#   BOTTOM : 미니언 격차 + 골드 격차 (ADC + SPT 합산)
# =====================================================================

# 포탑 골드 근사값 (Riot 타임라인에 선수별 포탑 골드 값이 없으므로 이벤트 개수로 근사)
PLATE_GOLD = 125    # 포탑 방패 1개
TURRET_GOLD = 250   # 포탑 1개


def get_top_target(match, timeline, data):
    # 1. 경험치 격차 : participantFrames 의 xp (누적값) 의 블루 - 레드
    # 2. 포탑 골드 격차 : TOP 라인에서 깬 방패/포탑 개수 x 골드
    #    TURRET_PLATE_DESTROYED, BUILDING_KILL 이벤트의 teamId 는 '잃은 팀' 이므로
    #    teamId == 200(레드가 잃음) 이면 블루가 골드를 얻은 것이다.
    #    방패는 미니언이 마무리하는 경우(killerId = 0)가 많아 선수가 아닌 팀 기준으로 센다.
    xpDiffBlue, xpDiffRed = [], []
    turretGoldDiffBlue, turretGoldDiffRed = [], []
    turretGoldBlue, turretGoldRed = 0, 0

    bluePid = data["TOP"]["blue"]["participantId"]
    redPid = data["TOP"]["red"]["participantId"]

    for frame in timeline["info"]["frames"]:
        pf = frame["participantFrames"]
        events = frame["events"]

        blue_xp = pf[str(bluePid)]["xp"]; red_xp = pf[str(redPid)]["xp"]
        xpDiffBlue.append(blue_xp - red_xp)
        xpDiffRed.append(red_xp - blue_xp)

        for event in events:
            if event.get("laneType") != "TOP_LANE":
                continue
            if event["type"] == "TURRET_PLATE_DESTROYED":
                gold = PLATE_GOLD
            elif event["type"] == "BUILDING_KILL" and event.get("buildingType") == "TOWER_BUILDING":
                gold = TURRET_GOLD
            else:
                continue
            if event["teamId"] == 200:
                turretGoldBlue += gold
            else:
                turretGoldRed += gold
        turretGoldDiffBlue.append(turretGoldBlue - turretGoldRed)
        turretGoldDiffRed.append(turretGoldRed - turretGoldBlue)

    data["TOP"]["blue"]["target"].update({"xpDiff": xpDiffBlue, "turretGoldDiff": turretGoldDiffBlue})
    data["TOP"]["red"]["target"].update({"xpDiff": xpDiffRed, "turretGoldDiff": turretGoldDiffRed})


def get_kill_assist_target(timeline, data, lane):
    # JUNGLE, MIDDLE 공통 : 킬 격차 + 어시스트 격차 (누적)
    # CHAMPION_KILL 이벤트에서 killerId 이면 킬, assistingParticipantIds 에 있으면 어시스트
    killDiffBlue, killDiffRed = [], []
    assistDiffBlue, assistDiffRed = [], []
    killBlue, killRed, assistBlue, assistRed = 0, 0, 0, 0

    bluePid = data[lane]["blue"]["participantId"]
    redPid = data[lane]["red"]["participantId"]

    for frame in timeline["info"]["frames"]:
        for event in frame["events"]:
            if event["type"] != "CHAMPION_KILL":
                continue
            if event["killerId"] == bluePid:
                killBlue += 1
            elif event["killerId"] == redPid:
                killRed += 1
            assists = event.get("assistingParticipantIds", [])
            if bluePid in assists:
                assistBlue += 1
            if redPid in assists:
                assistRed += 1
        killDiffBlue.append(killBlue - killRed)
        killDiffRed.append(killRed - killBlue)
        assistDiffBlue.append(assistBlue - assistRed)
        assistDiffRed.append(assistRed - assistBlue)

    data[lane]["blue"]["target"].update({"killDiff": killDiffBlue, "assistDiff": assistDiffBlue})
    data[lane]["red"]["target"].update({"killDiff": killDiffRed, "assistDiff": assistDiffRed})


def get_jungle_target(match, timeline, data):
    get_kill_assist_target(timeline, data, "JUNGLE")


def get_middle_target(match, timeline, data):
    get_kill_assist_target(timeline, data, "MIDDLE")


def get_bottom_target(match, timeline, data):
    # ADC + SPT 듀오 합산
    # 1. 미니언 격차 : minionsKilled (누적)
    # 2. 골드 격차   : totalGold (누적)
    # (와드 설치 격차는 단독 승패 예측 정확도가 57.4% 로 낮아 결합 타겟의 정확도를 떨어뜨리므로 제외)
    minionDiffBlue, minionDiffRed = [], []
    goldDiffBlue, goldDiffRed = [], []

    bluePids = list(data["BOTTOM"]["blue"]["participantId"].values())   # [ADC, SPT]
    redPids = list(data["BOTTOM"]["red"]["participantId"].values())

    for frame in timeline["info"]["frames"]:
        pf = frame["participantFrames"]

        blue_minion = sum(pf[str(p)]["minionsKilled"] for p in bluePids)
        red_minion = sum(pf[str(p)]["minionsKilled"] for p in redPids)
        minionDiffBlue.append(blue_minion - red_minion)
        minionDiffRed.append(red_minion - blue_minion)

        blue_gold = sum(pf[str(p)]["totalGold"] for p in bluePids)
        red_gold = sum(pf[str(p)]["totalGold"] for p in redPids)
        goldDiffBlue.append(blue_gold - red_gold)
        goldDiffRed.append(red_gold - blue_gold)

    data["BOTTOM"]["blue"]["target"].update({"minionDiff": minionDiffBlue, "goldDiff": goldDiffBlue})
    data["BOTTOM"]["red"]["target"].update({"minionDiff": minionDiffRed, "goldDiff": goldDiffRed})


# =====================================================================
# [추가] 데스 피처와 킬 타겟의 겹침 점검 (JUNGLE, MIDDLE)
# ---------------------------------------------------------------------
# 미드가 상대 미드를 처치하면 내 킬(타겟)과 상대 데스(피처)가 동시에 오른다.
# 데스 중 '상대 같은 라인에게 당한 데스' 의 비율을 세어, 데스 피처가 킬 타겟의 복사본이 아님을 확인한다.
# =====================================================================
def count_death_overlap(timeline, data):
    out = {}
    for lane in ["JUNGLE", "MIDDLE"]:
        b, r = data[lane]["blue"]["participantId"], data[lane]["red"]["participantId"]
        opp = {b: r, r: b}
        death, by_opp = 0, 0
        for frame in timeline["info"]["frames"]:
            for e in frame["events"]:
                if e["type"] == "CHAMPION_KILL" and e["victimId"] in opp:
                    death += 1
                    by_opp += e.get("killerId") == opp[e["victimId"]]
        out[f"{lane}_death"] = death
        out[f"{lane}_deathByOpp"] = by_opp
    return out


# =====================================================================
# [추가] 전체 데이터 처리
# ---------------------------------------------------------------------
# match_data 폴더의 match_N.json / timeline_N.json 짝(각 30경기)을 모두 읽는다.
# 경기 수가 많아(약 38,000경기) 경기별 data 딕셔너리를 모두 메모리에 들고 있지 않고,
# 경기마다 블루 관점의 (라인, 분) 행으로 펼쳐 저장한 뒤 버린다.
# 레드 관점 값은 블루 값의 부호만 바꾼 것이므로 저장하지 않는다 (tci_model.py 에서 생성).
# =====================================================================
LANES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM"]
DATA_DIR = "match_data"                       # match_N.json, timeline_N.json 이 있는 폴더
EXTRA_PAIRS = [("match_0.json", "timeline_0.json")]   # 폴더 밖 추가 파일 (없으면 무시)
MIN_DURATION_SEC = 20 * 60                    # 20분 미만 경기 제외
WORKERS = 4                                   # 동시에 처리할 파일 수 (메모리 부족 시 줄이기)
OUTPUT = "processed_data.pkl"


def find_pairs():
    pairs = []
    if os.path.isdir(DATA_DIR):
        for name in os.listdir(DATA_DIR):
            m = re.fullmatch(r"match_(\d+)\.json", name)
            if m:
                tl = os.path.join(DATA_DIR, f"timeline_{m.group(1)}.json")
                if os.path.exists(tl):
                    pairs.append((int(m.group(1)), os.path.join(DATA_DIR, name), tl))
    pairs.sort()
    pairs = [(mp, tp) for _, mp, tp in pairs]
    for mp, tp in EXTRA_PAIRS:
        if os.path.exists(mp) and os.path.exists(tp):
            pairs.insert(0, (mp, tp))
    return pairs


def is_valid(data):
    """5개 포지션이 모두 정상적으로 배정된 경기인지 확인 (다시하기·포지션 누락 경기 제외)"""
    for lane in ["TOP", "MIDDLE", "JUNGLE"]:
        for side in ["blue", "red"]:
            if "participantId" not in data[lane][side]:
                return False
    for side in ["blue", "red"]:
        if set(data["BOTTOM"][side]["participantId"]) != {"ADC", "SPT"}:
            return False
    return True


def flatten(data):
    """data 딕셔너리 → 블루 관점 (라인, 분) 행 리스트"""
    out = {lane: [] for lane in LANES}
    n = data["numFrames"]
    for lane in LANES:
        blue = data[lane]["blue"]
        red = data[lane]["red"]
        if lane == "BOTTOM":
            bc = int(any(v is not None for v in blue["challengerPuuid"].values()))
            rc = int(any(v is not None for v in red["challengerPuuid"].values()))
        else:
            bc = int(blue["challengerPuuid"] is not None)
            rc = int(red["challengerPuuid"] is not None)
        for t in range(n):
            row = {"matchId": data["matchId"], "minute": t, "isLast": int(t == n - 1),
                   "blueWin": int(data["blueWin"]), "blueChallenger": bc, "redChallenger": rc}
            for k, v in blue["feature"].items():
                row[k] = v[t]
            for k, v in blue["target"].items():
                row["T_" + k] = v[t]        # 타겟 구성 지표 (피처로 사용 금지)
            out[lane].append(row)
    return out


def process_pair(args):
    """match_N.json + timeline_N.json 한 쌍 처리 (별도 프로세스에서 실행)"""
    match_path, timeline_path, challengers = args
    with open(match_path, "r", encoding="utf-8") as f:
        matches = json.load(f)
    with open(timeline_path, "r", encoding="utf-8") as f:
        timelines = json.load(f)
    tl_by_id = {tl["metadata"]["matchId"]: tl for tl in timelines}

    rows = {lane: [] for lane in LANES}
    stat = {"total": len(matches), "short": 0, "invalid": 0, "error": 0, "used": 0,
            "JUNGLE_death": 0, "JUNGLE_deathByOpp": 0, "MIDDLE_death": 0, "MIDDLE_deathByOpp": 0}
    for m in matches:
        tl = tl_by_id.get(m["metadata"]["matchId"])
        if tl is None:
            stat["invalid"] += 1
            continue
        if m["info"]["gameDuration"] < MIN_DURATION_SEC:
            stat["short"] += 1
            continue
        try:
            data = get_default_match_data(m, tl, challengers)
            if not is_valid(data):
                stat["invalid"] += 1
                continue
            get_top_feature(m, tl, data)
            get_jungle_feature(m, tl, data)
            get_middle_feature(m, tl, data)
            get_bottom_feature(m, tl, data)
            get_top_target(m, tl, data)
            get_jungle_target(m, tl, data)
            get_middle_target(m, tl, data)
            get_bottom_target(m, tl, data)
            overlap = count_death_overlap(tl, data)
        except (KeyError, TypeError, IndexError):
            stat["error"] += 1
            continue
        for lane, r in flatten(data).items():
            rows[lane].extend(r)
        stat["used"] += 1
        for k, v in overlap.items():
            stat[k] += v

    # 메모리 절약 : 행 리스트를 float32 표로 변환해서 반환
    tables = {}
    for lane in LANES:
        df = pd.DataFrame(rows[lane])
        if len(df):
            num = df.columns.drop("matchId")
            df[num] = df[num].astype("float32")
        tables[lane] = df
    return tables, stat


# =====================================================================
# [추가] 타겟 정규화 : y = (각 지표 / 표준편차) 의 평균
# ---------------------------------------------------------------------
# 단위가 다른 지표(골드 수천 vs 킬 몇 개)를 같은 척도로 맞추기 위해 라인별 표준편차로 나눈 뒤 평균.
# 평균은 빼지 않는다 → y = 0 이 '대등', 레드의 y = - 블루의 y.
# =====================================================================
def compute_tci_target(tables):
    scale = {}
    for lane, df in tables.items():
        parts = [c for c in df.columns if c.startswith("T_")]
        scale[lane] = {c[2:]: float(df[c].std(ddof=0)) for c in parts}
        df["y"] = np.mean([df[c] / scale[lane][c[2:]] for c in parts], axis=0)
    return scale


# =====================================================================
# [추가] 타겟 검증 : y 의 부호가 실제 승패와 맞는가 (y == 0 인 경기는 제외)
# =====================================================================
def sign_accuracy(y, win):
    nz = y != 0
    return ((y[nz] > 0) == (win[nz] == 1)).mean()


def validate_target(tables):
    points = [5, 10, 15, 20, "종료"]
    n = tables["TOP"].matchId.nunique()
    print(f"\n[타겟 y 의 승패 예측 정확도]  (분석 매치 수 : {n:,})")
    print(f"{'단위':<8}" + "".join(f"{str(p):>10}" for p in points))
    result = {}
    for unit in LANES + ["TEAM"]:
        line, result[unit] = f"{unit:<8}", {}
        for p in points:
            if unit == "TEAM":
                parts = []
                for lane in LANES:
                    df = tables[lane]
                    sub = df[df.isLast == 1] if p == "종료" else df[df.minute == p]
                    parts.append(sub[["matchId", "y", "blueWin"]])
                sub = pd.concat(parts).groupby("matchId").agg(y=("y", "sum"), blueWin=("blueWin", "first"))
            else:
                df = tables[unit]
                sub = df[df.isLast == 1] if p == "종료" else df[df.minute == p]
            acc = sign_accuracy(sub.y.values, sub.blueWin.values)
            result[unit][str(p)] = acc
            line += f"{acc:>10.3f}"
        print(line)

    print("\n[타겟 구성 지표별 종료 시점 정확도]")
    for lane in LANES:
        last = tables[lane][tables[lane].isLast == 1]
        parts = [c for c in last.columns if c.startswith("T_")]
        print(f"  {lane:<7}: " + ", ".join(
            f"{c[2:]}={sign_accuracy(last[c].values, last.blueWin.values):.3f}" for c in parts))
    return result


# =====================================================================
# [추가] 피처 점검 : 피처와 y 의 상관계수, 피처끼리 |r| > 0.8 인 쌍
# =====================================================================
def check_features(tables):
    print("\n[피처와 타겟 y 의 상관계수]  (블루 관점, 0분 제외)")
    for lane, df in tables.items():
        sub = df[df.minute > 0]
        feats = [c for c in df.columns if c.endswith("Diff") and not c.startswith("T_")]
        corr = sub[feats].corrwith(sub["y"])
        print(f"  {lane:<7}: " + ", ".join(f"{k}={v:+.2f}" for k, v in corr.items()))
        cm = sub[feats].corr()
        for i, a in enumerate(feats):
            for b in feats[i + 1:]:
                if abs(cm.loc[a, b]) > 0.8:
                    print(f"           (중복 주의) {a} ~ {b} : r={cm.loc[a, b]:+.2f}")


def main():
    with open("puuid_challengers.txt", "r", encoding="utf-8") as f:
        challengers = set(f.read().splitlines())

    pairs = find_pairs()
    print(f"처리할 파일 쌍: {len(pairs)}개 (동시 처리 {WORKERS}개)")

    parts = {lane: [] for lane in LANES}
    stat = {"total": 0, "short": 0, "invalid": 0, "error": 0, "used": 0,
            "JUNGLE_death": 0, "JUNGLE_deathByOpp": 0, "MIDDLE_death": 0, "MIDDLE_deathByOpp": 0}
    seen = set()
    start = time.time()
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        jobs = [(mp, tp, challengers) for mp, tp in pairs]
        for i, (t, s) in enumerate(pool.map(process_pair, jobs), 1):
            for k in stat:
                stat[k] += s[k]
            if len(t["TOP"]):
                # 같은 경기가 여러 파일에 중복으로 들어 있으면 처음 것만 사용
                ids = set(t["TOP"].matchId.unique())
                dup = ids & seen
                seen |= ids
                stat["used"] -= len(dup)
                for lane in LANES:
                    df = t[lane]
                    parts[lane].append(df[~df.matchId.isin(dup)] if dup else df)
            if i % 50 == 0 or i == len(pairs):
                print(f"  {i}/{len(pairs)} 파일 완료, 사용 경기 {stat['used']:,} "
                      f"({(time.time() - start) / 60:.1f}분 경과)")

    print(f"\n전체 {stat['total']:,}경기 중 사용 {stat['used']:,}경기 "
          f"(20분 미만 {stat['short']:,}, 포지션 이상 {stat['invalid']:,}, 데이터 오류 {stat['error']:,} 제외)")

    tables = {lane: pd.concat(parts[lane], ignore_index=True) for lane in LANES}
    del parts
    for lane in LANES:   # 메모리 절약 : 경기 ID 를 범주형으로
        tables[lane]["matchId"] = tables[lane]["matchId"].astype("category")

    scale = compute_tci_target(tables)
    print("\n정규화에 사용한 표준편차:")
    for lane, s in scale.items():
        print(f"  {lane}: " + ", ".join(f"{k}={v:.2f}" for k, v in s.items()))

    validate_target(tables)
    check_features(tables)

    print("\n[데스 피처와 킬 타겟의 겹침]  데스 중 상대 같은 라인에게 당한 비율")
    for lane in ["JUNGLE", "MIDDLE"]:
        d, o = stat[f"{lane}_death"], stat[f"{lane}_deathByOpp"]
        print(f"  {lane:<7}: {o:,} / {d:,} = {o / d:.3f}")

    pd.to_pickle({"tables": tables, "scale": scale, "stat": stat}, OUTPUT)
    print(f"\n{OUTPUT} 저장 완료")


if __name__ == "__main__":
    main()