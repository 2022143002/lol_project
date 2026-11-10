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
#   MIDDLE : 미니언, 가한 피해, 받은 피해, 데스, 방패 파괴
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


def get_middle_feature(match, timeline, data):
    # 1. 미니언 격차    : minionsKilled (누적)
    # 2. 가한 피해 격차 : totalDamageDoneToChampions (누적)
    # 3. 받은 피해 격차 : totalDamageTaken (누적)
    # 4. 데스 격차      : victimId (이벤트 누적)
    # 5. 방패 파괴 격차 : TURRET_PLATE_DESTROYED 중 MID_LANE (팀 기준, 이벤트 누적)
    #                     teamId 는 '잃은 팀' 이므로 teamId == 200 이면 블루가 깬 것
    minionB, minionR, dmgDoneB, dmgDoneR, dmgTakenB, dmgTakenR = [], [], [], [], [], []
    deathB, deathR, plateB, plateR = [], [], [], []
    deathBlue, deathRed, plateBlue, plateRed = 0, 0, 0, 0

    bluePid = data["MIDDLE"]["blue"]["participantId"]
    redPid = data["MIDDLE"]["red"]["participantId"]

    for frame in timeline["info"]["frames"]:
        pf = frame["participantFrames"]
        b, r = pf[str(bluePid)], pf[str(redPid)]

        diff_pair(b["minionsKilled"], r["minionsKilled"], minionB, minionR)
        diff_pair(b["damageStats"]["totalDamageDoneToChampions"],
                  r["damageStats"]["totalDamageDoneToChampions"], dmgDoneB, dmgDoneR)
        diff_pair(b["damageStats"]["totalDamageTaken"],
                  r["damageStats"]["totalDamageTaken"], dmgTakenB, dmgTakenR)

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
        "deathDiff": deathB, "plateDiff": plateB})
    data["MIDDLE"]["red"]["feature"].update({
        "minionDiff": minionR, "dmgDoneDiff": dmgDoneR, "dmgTakenDiff": dmgTakenR,
        "deathDiff": deathR, "plateDiff": plateR})


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
#   BOTTOM : 미니언 격차 + 골드 격차 + 와드 설치 격차 (ADC + SPT 합산)
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
    # 3. 와드 격차   : WARD_PLACED 이벤트의 creatorId 기준 누적 (UNDEFINED 는 실제 와드가 아니므로 제외)
    minionDiffBlue, minionDiffRed = [], []
    goldDiffBlue, goldDiffRed = [], []
    wardDiffBlue, wardDiffRed = [], []
    wardBlue, wardRed = 0, 0

    bluePids = list(data["BOTTOM"]["blue"]["participantId"].values())   # [ADC, SPT]
    redPids = list(data["BOTTOM"]["red"]["participantId"].values())

    for frame in timeline["info"]["frames"]:
        pf = frame["participantFrames"]
        events = frame["events"]

        blue_minion = sum(pf[str(p)]["minionsKilled"] for p in bluePids)
        red_minion = sum(pf[str(p)]["minionsKilled"] for p in redPids)
        minionDiffBlue.append(blue_minion - red_minion)
        minionDiffRed.append(red_minion - blue_minion)

        blue_gold = sum(pf[str(p)]["totalGold"] for p in bluePids)
        red_gold = sum(pf[str(p)]["totalGold"] for p in redPids)
        goldDiffBlue.append(blue_gold - red_gold)
        goldDiffRed.append(red_gold - blue_gold)

        for event in events:
            if event["type"] == "WARD_PLACED" and event.get("wardType") != "UNDEFINED":
                if event["creatorId"] in bluePids:
                    wardBlue += 1
                elif event["creatorId"] in redPids:
                    wardRed += 1
        wardDiffBlue.append(wardBlue - wardRed)
        wardDiffRed.append(wardRed - wardBlue)

    data["BOTTOM"]["blue"]["target"].update(
        {"minionDiff": minionDiffBlue, "goldDiff": goldDiffBlue, "wardDiff": wardDiffBlue})
    data["BOTTOM"]["red"]["target"].update(
        {"minionDiff": minionDiffRed, "goldDiff": goldDiffRed, "wardDiff": wardDiffRed})


# =====================================================================
# [추가] 타겟 정규화 : y = (각 지표 / 표준편차) 의 평균
# ---------------------------------------------------------------------
# 단위가 다른 지표(골드 수천 vs 킬 몇 개)를 그대로 더하면 큰 쪽이 전부를 차지하므로
# 라인별 표준편차로 나눠 같은 척도로 맞춘 뒤 평균한다.
# 평균은 빼지 않는다 → y = 0 이 '대등', 레드의 y = - 블루의 y 가 유지된다.
# 표준편차는 전체 매치 x 전체 분의 블루 관점 값으로 계산한다.
# =====================================================================
def std(values):
    m = sum(values) / len(values)
    return (sum((v - m) ** 2 for v in values) / len(values)) ** 0.5


def compute_tci_target(data_list):
    lanes = ["TOP", "MIDDLE", "JUNGLE", "BOTTOM"]
    scale = {}
    for lane in lanes:
        names = list(data_list[0][lane]["blue"]["target"].keys())
        scale[lane] = {}
        for name in names:
            all_values = []
            for data in data_list:
                all_values += data[lane]["blue"]["target"][name]
            scale[lane][name] = std(all_values)

        for data in data_list:
            for side in ["blue", "red"]:
                target = data[lane][side]["target"]
                n = len(target[names[0]])
                y = []
                for t in range(n):
                    z = [target[name][t] / scale[lane][name] for name in names]
                    y.append(sum(z) / len(z))
                data[lane][side]["y"] = y
    return scale


# =====================================================================
# [추가] 타겟 검증 : y 의 부호가 실제 승패와 맞는가
# ---------------------------------------------------------------------
# 블루 관점 y > 0 이면 블루 승리로 예측. y == 0 (대등) 인 경기는 판정에서 제외.
# 라인 단위 + 팀 단위(4개 라인 y 의 합) 를 5/10/15/20분, 종료 시점에서 확인한다.
# =====================================================================
def validate_target(data_list):
    lanes = ["TOP", "MIDDLE", "JUNGLE", "BOTTOM"]
    points = [5, 10, 15, 20, "종료"]
    print(f"\n[타겟 y 의 승패 예측 정확도]  (분석 매치 수 : {len(data_list)})")
    print(f"{'단위':<8}" + "".join(f"{str(p):>10}" for p in points))

    for unit in lanes + ["TEAM"]:
        line = f"{unit:<8}"
        for p in points:
            hit, total = 0, 0
            for data in data_list:
                t = data["numFrames"] - 1 if p == "종료" else p
                if unit == "TEAM":
                    y = sum(data[lane]["blue"]["y"][t] for lane in lanes)
                else:
                    y = data[unit]["blue"]["y"][t]
                if y == 0:
                    continue
                total += 1
                hit += (y > 0) == data["blueWin"]
            line += f"{hit / total:>10.3f}"
        print(line)


# =====================================================================
# [추가] 피처 점검 : 각 피처가 타겟 y 와 얼마나 관련 있는가
# ---------------------------------------------------------------------
# (1) 상관계수 : 전체 매치 x 전체 분(0분 제외)의 피처 값과 y 의 피어슨 상관
# (2) 피처끼리의 상관 : |r| > 0.8 이면 같은 정보를 중복으로 담고 있다고 보고 출력
# =====================================================================
def pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = sum((x - ma) ** 2 for x in a) ** 0.5
    vb = sum((y - mb) ** 2 for y in b) ** 0.5
    return cov / (va * vb) if va > 0 and vb > 0 else 0.0


def check_features(data_list):
    lanes = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM"]
    print("\n[피처와 타겟 y 의 상관계수]  (블루 관점, 0분 제외)")
    for lane in lanes:
        names = list(data_list[0][lane]["blue"]["feature"].keys())
        values = {name: [] for name in names}
        ys = []
        for data in data_list:
            blue = data[lane]["blue"]
            for name in names:
                values[name] += blue["feature"][name][1:]
            ys += blue["y"][1:]
        corr = {name: pearson(values[name], ys) for name in names}
        print(f"  {lane:<7}: " + ", ".join(f"{k}={v:+.2f}" for k, v in corr.items()))
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                r = pearson(values[a], values[b])
                if abs(r) > 0.8:
                    print(f"           (중복 주의) {a} ~ {b} : r={r:+.2f}")


def main():
    with open("match_0.json", "r", encoding="utf-8") as f:
        match = json.load(f)
    with open("timeline_0.json", "r", encoding="utf-8") as f:
        timeline = json.load(f)
    with open("puuid_challengers.txt", "r", encoding="utf-8") as f:
        challengers = f.read().splitlines()

    # 전체 매치 반복 (20분 미만 경기는 제외)
    data_list = []
    for m, tl in zip(match, timeline):
        if m["info"]["gameDuration"] < 20 * 60:
            continue
        data = get_default_match_data(m, tl, challengers)

        # 라인별 피처(x) 추출
        get_top_feature(m, tl, data)
        get_jungle_feature(m, tl, data)
        get_middle_feature(m, tl, data)
        get_bottom_feature(m, tl, data)

        # 라인별 타겟(y) 추출
        get_top_target(m, tl, data)
        get_jungle_target(m, tl, data)
        get_middle_target(m, tl, data)
        get_bottom_target(m, tl, data)
        data_list.append(data)

    # 타겟 정규화 및 검증
    scale = compute_tci_target(data_list)
    print("정규화에 사용한 표준편차:")
    for lane, s in scale.items():
        print(f"  {lane}: " + ", ".join(f"{k}={v:.2f}" for k, v in s.items()))
    validate_target(data_list)

    # 피처 점검
    check_features(data_list)

    # 결과 저장
    with open("processed_data.json", "w", encoding="utf-8") as f:
        json.dump(data_list, f, ensure_ascii=False)
    print("\nprocessed_data.json 저장 완료")


if __name__ == "__main__":
    main()
