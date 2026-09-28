import math

from std_msgs.msg import Float64MultiArray


DETECTION_DIAGNOSTICS_TOPIC = '/perception/detection_diagnostics'

DETECTION_DIAGNOSTIC_FIELDS = (
    'stamp',
    'detected',
    'u_mid',
    'v_mid',
    'yaw_est',
    'u_red',
    'v_red',
    'red_area_px',
    'u_blue',
    'v_blue',
    'blue_area_px',
    'separation_px',
    'border_margin_px',
    'yolo_score_raw',
    'yolo_score_selected',
    'yolo_detected_after_threshold',
    'yolo_best_class_id',
    'yolo_target_candidate_count',
    'bbox_area_px',
    'bbox_xmin',
    'bbox_ymin',
    'bbox_xmax',
    'bbox_ymax',
    'logit_margin',
    'class_entropy',
    'mask_area_px',
    'mask_bottom_u',
    'mask_bottom_v',
    'mask_used',
    'mask_polygon_points',
    'confidence_logit',
    'mask_compactness',
    'mask_border_frac',
    'mask_score',
    'selected_pixel_source_code',
    'yolo_inference_ms',
    'detector_callback_ms',
    # Runtime causality ledger.  These are ROS-clock stamps in seconds and
    # preserve the delayed-measurement chain: image -> detector -> correction.
    'yolo_receive_stamp',
    'yolo_start_stamp',
    'yolo_finish_stamp',
    'yolo_publish_stamp',
    'yolo_latency_s',
    'frame_age_at_publish_s',
)

_FIELD_INDEX = {
    name: idx for idx, name in enumerate(DETECTION_DIAGNOSTIC_FIELDS)
}


def diagnostics_message(
    *,
    stamp,
    detected,
    u_mid,
    v_mid,
    yaw_est,
    u_red,
    v_red,
    red_area_px,
    u_blue,
    v_blue,
    blue_area_px,
    separation_px,
    border_margin_px,
    yolo_score_raw=math.nan,
    yolo_score_selected=math.nan,
    yolo_detected_after_threshold=math.nan,
    yolo_best_class_id=math.nan,
    yolo_target_candidate_count=math.nan,
    bbox_area_px=math.nan,
    bbox_xmin=math.nan,
    bbox_ymin=math.nan,
    bbox_xmax=math.nan,
    bbox_ymax=math.nan,
    logit_margin=math.nan,
    class_entropy=math.nan,
    mask_area_px=math.nan,
    mask_bottom_u=math.nan,
    mask_bottom_v=math.nan,
    mask_used=math.nan,
    mask_polygon_points=math.nan,
    confidence_logit=math.nan,
    mask_compactness=math.nan,
    mask_border_frac=math.nan,
    mask_score=math.nan,
    selected_pixel_source_code=math.nan,
    yolo_inference_ms=math.nan,
    detector_callback_ms=math.nan,
    yolo_receive_stamp=math.nan,
    yolo_start_stamp=math.nan,
    yolo_finish_stamp=math.nan,
    yolo_publish_stamp=math.nan,
    yolo_latency_s=math.nan,
    frame_age_at_publish_s=math.nan,
):
    msg = Float64MultiArray()
    msg.data = [
        float(stamp),
        1.0 if detected else 0.0,
        float(u_mid),
        float(v_mid),
        float(yaw_est),
        float(u_red),
        float(v_red),
        float(red_area_px),
        float(u_blue),
        float(v_blue),
        float(blue_area_px),
        float(separation_px),
        float(border_margin_px),
        float(yolo_score_raw),
        float(yolo_score_selected),
        float(yolo_detected_after_threshold),
        float(yolo_best_class_id),
        float(yolo_target_candidate_count),
        float(bbox_area_px),
        float(bbox_xmin),
        float(bbox_ymin),
        float(bbox_xmax),
        float(bbox_ymax),
        float(logit_margin),
        float(class_entropy),
        float(mask_area_px),
        float(mask_bottom_u),
        float(mask_bottom_v),
        float(mask_used),
        float(mask_polygon_points),
        float(confidence_logit),
        float(mask_compactness),
        float(mask_border_frac),
        float(mask_score),
        float(selected_pixel_source_code),
        float(yolo_inference_ms),
        float(detector_callback_ms),
        float(yolo_receive_stamp),
        float(yolo_start_stamp),
        float(yolo_finish_stamp),
        float(yolo_publish_stamp),
        float(yolo_latency_s),
        float(frame_age_at_publish_s),
    ]
    return msg


def diagnostics_from_message(msg: Float64MultiArray):
    data = list(msg.data)
    if len(data) < len(DETECTION_DIAGNOSTIC_FIELDS):
        data.extend([math.nan] * (len(DETECTION_DIAGNOSTIC_FIELDS) - len(data)))

    values = {
        name: float(data[idx]) for name, idx in _FIELD_INDEX.items()
    }
    values['detected'] = bool(values['detected'] >= 0.5)
    return values
