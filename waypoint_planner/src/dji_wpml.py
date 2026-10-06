"""Build a DJI WPML .kmz (template.kml + waylines.wpml) for a lawn-mower mission.

Each flight line = 2 waypoints (start, end). One action group per line with trigger
`multipleDistance` (or `multipleTiming`) spans start..end, so the aircraft shoots automatically in between.
"""
import re
import time
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

NS = 'xmlns="http://www.opengis.net/kml/2.2" xmlns:wpml="http://www.dji.com/wpmz/1.0.6"'


def enums_from_reference(kmz_path):
    with zipfile.ZipFile(kmz_path) as z:
        name = next(n for n in z.namelist() if n.endswith("template.kml") or n.endswith("waylines.wpml"))
        x = z.read(name).decode("utf8")
    g = lambda tag: re.search(rf"<wpml:{tag}>([^<]*)</wpml:{tag}>", x).group(1)
    return dict(drone_enum=g("droneEnumValue"), drone_sub_enum=g("droneSubEnumValue"),
                payload_enum=g("payloadEnumValue"), payload_position=g("payloadPositionIndex"),
                payload_sub_enum=(re.search(r"<wpml:payloadSubEnumValue>([^<]*)<", x) or [0, 0])[1])


def _mission_config(d):
    return f"""  <wpml:missionConfig>
    <wpml:flyToWaylineMode>{d['fly_to_wayline_mode']}</wpml:flyToWaylineMode>
    <wpml:finishAction>{d['finish_action']}</wpml:finishAction>
    <wpml:exitOnRCLost>{d['exit_on_rc_lost']}</wpml:exitOnRCLost>
    <wpml:executeRCLostAction>{d['rc_action']}</wpml:executeRCLostAction>
    <wpml:takeOffSecurityHeight>{d['take_off_security_height_m']}</wpml:takeOffSecurityHeight>
    <wpml:globalTransitionalSpeed>{d['transitional_speed_ms']}</wpml:globalTransitionalSpeed>
    <wpml:globalRTHHeight>{d['rth_height_m']}</wpml:globalRTHHeight>
    <wpml:droneInfo>
      <wpml:droneEnumValue>{d['drone_enum']}</wpml:droneEnumValue>
      <wpml:droneSubEnumValue>{d['drone_sub_enum']}</wpml:droneSubEnumValue>
    </wpml:droneInfo>
    <wpml:waylineAvoidLimitAreaMode>1</wpml:waylineAvoidLimitAreaMode>
    <wpml:payloadInfo>
      <wpml:payloadEnumValue>{d['payload_enum']}</wpml:payloadEnumValue>
      <wpml:payloadSubEnumValue>{d['payload_sub_enum']}</wpml:payloadSubEnumValue>
      <wpml:payloadPositionIndex>{d['payload_position']}</wpml:payloadPositionIndex>
    </wpml:payloadInfo>
  </wpml:missionConfig>"""


def _start_actions(p):
    """Gimbal nadir + tele zoom, once at mission start (waylines.wpml only, as in Pilot 2 exports)."""
    ppos = p["payload_position"]
    zoom = ""
    if p.get("zoom_focal_length_mm"):
        zoom = f"""<wpml:action><wpml:actionId>1</wpml:actionId><wpml:actionActuatorFunc>zoom</wpml:actionActuatorFunc>
        <wpml:actionActuatorFuncParam><wpml:focalLength>{p['zoom_focal_length_mm']}</wpml:focalLength>
        <wpml:payloadPositionIndex>{ppos}</wpml:payloadPositionIndex></wpml:actionActuatorFuncParam></wpml:action>"""
    return f"""    <wpml:startActionGroup>
      <wpml:action><wpml:actionId>0</wpml:actionId><wpml:actionActuatorFunc>gimbalRotate</wpml:actionActuatorFunc>
        <wpml:actionActuatorFuncParam><wpml:gimbalHeadingYawBase>aircraft</wpml:gimbalHeadingYawBase>
        <wpml:gimbalRotateMode>absoluteAngle</wpml:gimbalRotateMode>
        <wpml:gimbalPitchRotateEnable>1</wpml:gimbalPitchRotateEnable><wpml:gimbalPitchRotateAngle>{p['gimbal_pitch_deg']}</wpml:gimbalPitchRotateAngle>
        <wpml:gimbalRollRotateEnable>0</wpml:gimbalRollRotateEnable><wpml:gimbalRollRotateAngle>0</wpml:gimbalRollRotateAngle>
        <wpml:gimbalYawRotateEnable>1</wpml:gimbalYawRotateEnable><wpml:gimbalYawRotateAngle>0</wpml:gimbalYawRotateAngle>
        <wpml:gimbalRotateTimeEnable>1</wpml:gimbalRotateTimeEnable><wpml:gimbalRotateTime>2</wpml:gimbalRotateTime>
        <wpml:payloadPositionIndex>{ppos}</wpml:payloadPositionIndex></wpml:actionActuatorFuncParam></wpml:action>
      {zoom}
    </wpml:startActionGroup>"""


def _actions(wp_index, lines_idx, p):
    """Photo action group for the line that starts at waypoint `wp_index` (shoots start..end automatically)."""
    out, ppos = [], p["payload_position"]
    for k, (st, e) in enumerate(lines_idx):
        if st != wp_index:
            continue
        out.append(f"""<wpml:actionGroup><wpml:actionGroupId>{k}</wpml:actionGroupId>
        <wpml:actionGroupStartIndex>{st}</wpml:actionGroupStartIndex><wpml:actionGroupEndIndex>{e}</wpml:actionGroupEndIndex>
        <wpml:actionGroupMode>sequence</wpml:actionGroupMode>
        <wpml:actionTrigger><wpml:actionTriggerType>{p['photo_trigger']}</wpml:actionTriggerType><wpml:actionTriggerParam>{p['trigger_param']:.2f}</wpml:actionTriggerParam></wpml:actionTrigger>
        <wpml:action><wpml:actionId>0</wpml:actionId><wpml:actionActuatorFunc>takePhoto</wpml:actionActuatorFunc>
          <wpml:actionActuatorFuncParam><wpml:fileSuffix>L{k:03d}</wpml:fileSuffix>
          <wpml:payloadPositionIndex>{ppos}</wpml:payloadPositionIndex>
          <wpml:useGlobalPayloadLensIndex>0</wpml:useGlobalPayloadLensIndex>
          <wpml:payloadLensIndex>{p['image_format']}</wpml:payloadLensIndex></wpml:actionActuatorFuncParam></wpml:action></wpml:actionGroup>""")
    return "\n      ".join(out)


def build_kmz(path, wps, line_idx, height_m, speed_ms, p, dist=0.0, dur=0.0):
    """wps: list of (lon, lat). line_idx: list of (start_wp, end_wp). p: dict of dji params + trigger_param."""
    now = int(time.time() * 1000)
    head = f'<?xml version="1.0" encoding="UTF-8"?>\n<kml {NS}>\n<Document>\n'
    turn = "toPointAndStopWithDiscontinuityCurvature"

    pm_t, pm_w = [], []
    for i, (lon, lat) in enumerate(wps):
        act = _actions(i, line_idx, p)
        pm_t.append(f"""    <Placemark>
      <Point><coordinates>{lon:.8f},{lat:.8f}</coordinates></Point>
      <wpml:index>{i}</wpml:index>
      <wpml:ellipsoidHeight>{height_m}</wpml:ellipsoidHeight>
      <wpml:height>{height_m}</wpml:height>
      <wpml:useGlobalHeight>1</wpml:useGlobalHeight>
      <wpml:useGlobalSpeed>1</wpml:useGlobalSpeed>
      <wpml:useGlobalHeadingParam>1</wpml:useGlobalHeadingParam>
      <wpml:useGlobalTurnParam>1</wpml:useGlobalTurnParam>
      <wpml:gimbalPitchAngle>{p['gimbal_pitch_deg']}</wpml:gimbalPitchAngle>
      {act}
    </Placemark>""")
        pm_w.append(f"""    <Placemark>
      <Point><coordinates>{lon:.8f},{lat:.8f}</coordinates></Point>
      <wpml:index>{i}</wpml:index>
      <wpml:executeHeight>{height_m}</wpml:executeHeight>
      <wpml:waypointSpeed>{speed_ms}</wpml:waypointSpeed>
      <wpml:waypointHeadingParam>
        <wpml:waypointHeadingMode>{p['heading_mode']}</wpml:waypointHeadingMode>
        <wpml:waypointHeadingAngle>{p['heading_deg']:g}</wpml:waypointHeadingAngle>
        <wpml:waypointPoiPoint>0.000000,0.000000,0.000000</wpml:waypointPoiPoint>
        <wpml:waypointHeadingAngleEnable>{1 if p['heading_mode'] == 'fixed' else 0}</wpml:waypointHeadingAngleEnable>
        <wpml:waypointHeadingPathMode>followBadArc</wpml:waypointHeadingPathMode>
        <wpml:waypointHeadingPoiIndex>0</wpml:waypointHeadingPoiIndex>
      </wpml:waypointHeadingParam>
      <wpml:waypointTurnParam>
        <wpml:waypointTurnMode>{turn}</wpml:waypointTurnMode>
        <wpml:waypointTurnDampingDist>0</wpml:waypointTurnDampingDist>
      </wpml:waypointTurnParam>
      <wpml:useStraightLine>1</wpml:useStraightLine>
      {act}
      <wpml:waypointGimbalHeadingParam>
        <wpml:waypointGimbalPitchAngle>{p['gimbal_pitch_deg']}</wpml:waypointGimbalPitchAngle>
        <wpml:waypointGimbalYawAngle>0</wpml:waypointGimbalYawAngle>
      </wpml:waypointGimbalHeadingParam>
      <wpml:isRisky>0</wpml:isRisky>
      <wpml:waypointWorkType>0</wpml:waypointWorkType>
    </Placemark>""")

    template = head + f"""  <wpml:author>waypoint_planner</wpml:author>
  <wpml:createTime>{now}</wpml:createTime>
  <wpml:updateTime>{now}</wpml:updateTime>
{_mission_config(p)}
  <Folder>
    <wpml:templateType>waypoint</wpml:templateType>
    <wpml:templateId>0</wpml:templateId>
    <wpml:waylineCoordinateSysParam>
      <wpml:coordinateMode>WGS84</wpml:coordinateMode>
      <wpml:heightMode>relativeToStartPoint</wpml:heightMode>
      <wpml:positioningType>{p["positioning_type"]}</wpml:positioningType>
    </wpml:waylineCoordinateSysParam>
    <wpml:autoFlightSpeed>{speed_ms}</wpml:autoFlightSpeed>
    <wpml:globalHeight>{height_m}</wpml:globalHeight>
    <wpml:caliFlightEnable>0</wpml:caliFlightEnable>
    <wpml:gimbalPitchMode>usePointSetting</wpml:gimbalPitchMode>
    <wpml:globalWaypointHeadingParam>
      <wpml:waypointHeadingMode>{p['heading_mode']}</wpml:waypointHeadingMode>
      <wpml:waypointHeadingAngle>{p['heading_deg']:g}</wpml:waypointHeadingAngle>
      <wpml:waypointPoiPoint>0.000000,0.000000,0.000000</wpml:waypointPoiPoint>
      <wpml:waypointHeadingPoiIndex>0</wpml:waypointHeadingPoiIndex>
    </wpml:globalWaypointHeadingParam>
    <wpml:globalWaypointTurnMode>{turn}</wpml:globalWaypointTurnMode>
    <wpml:globalUseStraightLine>1</wpml:globalUseStraightLine>
{chr(10).join(pm_t)}
    <wpml:payloadParam>
      <wpml:payloadPositionIndex>{p['payload_position']}</wpml:payloadPositionIndex>
      <wpml:focusMode>firstPoint</wpml:focusMode>
      <wpml:meteringMode>average</wpml:meteringMode>
      <wpml:dewarpingEnable>0</wpml:dewarpingEnable>
      <wpml:returnMode>singleReturnFirst</wpml:returnMode>
      <wpml:samplingRate>240000</wpml:samplingRate>
      <wpml:scanningMode>nonRepetitive</wpml:scanningMode>
      <wpml:modelColoringEnable>0</wpml:modelColoringEnable>
      <wpml:imageFormat>{p['image_format']}</wpml:imageFormat>
    </wpml:payloadParam>
  </Folder>
</Document>
</kml>
"""
    waylines = head + f"""{_mission_config(p)}
  <Folder>
    <wpml:templateId>0</wpml:templateId>
    <wpml:executeHeightMode>relativeToStartPoint</wpml:executeHeightMode>
    <wpml:waylineId>0</wpml:waylineId>
    <wpml:distance>{dist:.0f}</wpml:distance>
    <wpml:duration>{dur:.0f}</wpml:duration>
    <wpml:autoFlightSpeed>{speed_ms}</wpml:autoFlightSpeed>
{_start_actions(p)}
    <wpml:realTimeFollowSurfaceByFov>0</wpml:realTimeFollowSurfaceByFov>
{chr(10).join(pm_w)}
  </Folder>
</Document>
</kml>
"""
    path = Path(path)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("wpmz/template.kml", template)
        z.writestr("wpmz/waylines.wpml", waylines)
        z.writestr("wpmz/res/", "")
    return path
