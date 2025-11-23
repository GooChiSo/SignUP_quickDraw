#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
실시간 키포인트 기반 수어 퀴즈 모드 (QuickDraw 스타일, s로 시작, 카테고리 1개 전용)

- CATEGORY = "팔 때리다 당황" 같이 문장 폴더 이름을 입력하면
  → ["팔", "때리다", "당황"] 한 문장만 퀴즈로 사용

- 웹캠 → MediaPipe Holistic → 3-features(152, 토르소 정규화) → GRU(v2)+Attention
- 길이 정책: 최근 프레임을 sliding window로 유지 (최대 90), 최소 30프레임 이상일 때만 예측
- 퀴즈 정책:
    · CATEGORY에서 단어들을 split해서 문장으로 사용
    · 현재 타겟 단어의 보정 확률 p_target'이 PASS_THRESHOLD 이상인 결과가
      최근 HIT_WINDOW 번 예측 중 HIT_REQUIRED 번 이상 나오면 → 자동으로 다음 단어로 넘어감
- UI:
    · 상단: 현재 문장 / 단어 진행 상황 표시
    · 중간: 현재 단어 점수(Perfect/OK/Not Bad/Bad + %) 표시
    · 하단: 키 안내 (s/r/n/q)

키:
    s : 퀴즈 시작 (대기 → 진행, sequence/hit_history 초기화)
    r : 현재 문장을 처음부터 다시
    n : 현재 단어 스킵하고 다음 단어로
    q : 종료
"""

import os
import json
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F
from PIL import ImageFont, ImageDraw, Image
import pickle

from model_v2 import KeypointGRUModelV2

# =========================
# 0) 설정
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _load_font():
    candidates = [
        "NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/Library/Fonts/AppleGothic.ttf",
        "C:/Windows/Fonts/malgun.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, 12)
            except Exception:
                pass
    return ImageFont.load_default()

FONT = _load_font()

# 길이 정책 (슬라이딩 윈도우)
FRAME_MIN    = 30   # 최소 이 정도 이상일 때만 모델에 넣음
FRAME_TARGET = 90   # 버퍼 최대 길이 (최근 90프레임까지만 유지)

# 경로/설정
MODEL_DIR = "models_sentences_v2"  # 문장용 모델 저장 경로에 맞게 수정
USE_CANON = True  # 토르소 정규화 사용 여부

# 등급 기준 (보정 확률 p_target')
THR_BAD      = 0.55   # 미만이면 Bad
THR_NOTBAD   = 0.55   # 이상
THR_OK       = 0.70   # 이상
THR_PERFECT  = 0.95   # 이상

# 퀴즈용 패스 기준
PASS_THRESHOLD = 0.80   # 이 이상이면 "맞은 것 같다"로 간주
HIT_WINDOW     = 5      # 최근 5번 예측 중
HIT_REQUIRED   = 3      # 3번 이상 PASS_THRESHOLD를 넘으면 단어 통과

PRED_INTERVAL  = 3      # N프레임마다 한 번씩 예측


# =========================
# 1) 카테고리 선택/모델 & 보정 로드
# =========================
CATEGORY = input("퀴즈에 사용할 카테고리(문장 이름)를 입력하세요: ").strip()
label_map_path = os.path.join(MODEL_DIR, f"{CATEGORY}_label_map.pkl")
model_path     = os.path.join(MODEL_DIR, f"{CATEGORY}_model.pth")
calib_path     = os.path.join(MODEL_DIR, f"{CATEGORY}_calib.json")

if not os.path.exists(label_map_path) or not os.path.exists(model_path):
    print(f"[ERR] 모델 또는 라벨맵 없음: {label_map_path}, {model_path}")
    raise SystemExit(2)
if not os.path.exists(calib_path):
    print(f"[ERR] 보정 파일 없음: {calib_path}")
    raise SystemExit(2)

with open(label_map_path, "rb") as f:
    label_map = pickle.load(f)
idx_to_label = {v: k for k, v in label_map.items()}

with open(calib_path, "r", encoding="utf-8") as f:
    calib = json.load(f)
T     = float(calib["temperature"])
BETA  = float(calib["beta"])

num_classes = len(label_map)
model = KeypointGRUModelV2(input_dim=152, attn_dim=146, num_classes=num_classes).to(DEVICE)
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.eval()

print(f"[{CATEGORY}] v2 모델 + Calibration 로드 완료 — classes: {len(label_map)}  device: {DEVICE}")
print(f"  Temperature T={T:.3f}, Beta={BETA:.3f}")
print(f"  등급 기준: Bad<{THR_BAD*100:.0f}% < NotBad/OK < {THR_PERFECT*100:.0f}%≤Perfect")

# =========================
# 2) 카테고리 이름에서 문장 한 개 자동 구성
# =========================
"""
예:
  CATEGORY = "팔 때리다 당황"
→ sentence_words = ["팔", "때리다", "당황"]
"""

sentence_words = CATEGORY.split()
SENTENCES = [sentence_words]   # 문장 1개만 사용

# label_map에 없는 단어가 있으면 경고
for w in sentence_words:
    if w not in label_map:
        print(f"[WARN] 문장 단어 '{w}' 가 label_map에 없습니다. (모델에 없는 라벨일 수 있음)")

current_sentence_idx = 0  # 항상 0
current_word_idx     = 0  # 0 ~ len(sentence_words)-1

def get_current_target():
    label = sentence_words[current_word_idx]
    idx   = label_map.get(label, None)
    return label, idx

current_target_label, current_target_idx = get_current_target()

print(f"문장: {' '.join(sentence_words)}")
print(f"초기 단어: 1/{len(sentence_words)} ('{current_target_label}')")
print("키: s=퀴즈 시작, r=문장 리셋, n=현재 단어 스킵, q=종료")


# =========================
# 3) MediaPipe 초기화
# =========================
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1
)

# =========================
# 4) 토르소 정규화 / 3-features 추출 유틸
# =========================
POSE_L_SHOULDER = 11
POSE_R_SHOULDER = 12
POSE_L_HIP      = 23
POSE_R_HIP      = 24

def safe_unit(v, eps=1e-8):
    n = np.linalg.norm(v)
    if n < eps:
        return v * 0.0
    return v / n

def build_torso_frame(pose_33x4):
    try:
        l_sh = pose_33x4[POSE_L_SHOULDER, :3]
        r_sh = pose_33x4[POSE_R_SHOULDER, :3]
        l_hp = pose_33x4[POSE_L_HIP, :3]
        r_hp = pose_33x4[POSE_R_HIP, :3]
    except Exception:
        return np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32), 1.0, False

    if np.all(l_sh == 0) or np.all(r_sh == 0) or np.all(l_hp == 0) or np.all(r_hp == 0):
        return np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32), 1.0, False

    mid_hip = 0.5 * (l_hp + r_hp)
    mid_sh  = 0.5 * (l_sh + r_sh)

    x_axis = safe_unit(r_sh - l_sh)
    y_axis = safe_unit(mid_sh - mid_hip)
    y_axis = safe_unit(y_axis - np.dot(y_axis, x_axis) * x_axis)
    z_axis = safe_unit(np.cross(x_axis, y_axis))
    x_axis = safe_unit(np.cross(y_axis, z_axis))
    y_axis = safe_unit(np.cross(z_axis, x_axis))
    R = np.stack([x_axis, y_axis, z_axis], axis=1).astype(np.float32)
    scale = max(np.linalg.norm(r_sh - l_sh), 1e-3)
    return R, mid_hip.astype(np.float32), float(scale), True

def canonicalize(pose_33x4, face_468x3, lh_21x3, rh_21x3):
    R, origin, scale, ok = build_torso_frame(pose_33x4)
    if not ok:
        return pose_33x4, face_468x3, lh_21x3, rh_21x3

    def xf(P):
        if P.size == 0:
            return P
        Q = (P - origin[None, :]) @ R
        return Q / scale

    pose_c = xf(pose_33x4[:, :3])
    face_c = xf(face_468x3)
    lh_c   = xf(lh_21x3)
    rh_c   = xf(rh_21x3)
    return pose_c, face_c, lh_c, rh_c

def landmarks_to_np(results):
    pose_33x4  = np.zeros((33, 4), dtype=np.float32)
    face_468x3 = np.zeros((468, 3), dtype=np.float32)
    lh_21x3    = np.zeros((21, 3), dtype=np.float32)
    rh_21x3    = np.zeros((21, 3), dtype=np.float32)

    if results.pose_landmarks:
        for i, lm in enumerate(results.pose_landmarks.landmark):
            if i < 33:
                pose_33x4[i, 0] = lm.x
                pose_33x4[i, 1] = lm.y
                pose_33x4[i, 2] = lm.z
                pose_33x4[i, 3] = lm.visibility

    if results.face_landmarks:
        for i, lm in enumerate(results.face_landmarks.landmark):
            if i < 468:
                face_468x3[i, 0] = lm.x
                face_468x3[i, 1] = lm.y
                face_468x3[i, 2] = lm.z

    if results.left_hand_landmarks:
        for i, lm in enumerate(results.left_hand_landmarks.landmark):
            if i < 21:
                lh_21x3[i, 0] = lm.x
                lh_21x3[i, 1] = lm.y
                lh_21x3[i, 2] = lm.z

    if results.right_hand_landmarks:
        for i, lm in enumerate(results.right_hand_landmarks.landmark):
            if i < 21:
                rh_21x3[i, 0] = lm.x
                rh_21x3[i, 1] = lm.y
                rh_21x3[i, 2] = lm.z

    return pose_33x4, face_468x3, lh_21x3, rh_21x3

def calculate_relative_hand_coords(lh_21x3, rh_21x3):
    def rel_hand(hand):
        if np.all(hand == 0):
            return np.zeros_like(hand)
        wrist = hand[0:1, :]
        return hand - wrist

    lh_rel = rel_hand(lh_21x3)
    rh_rel = rel_hand(rh_21x3)
    return lh_rel.reshape(-1), rh_rel.reshape(-1)

def calculate_finger_angles(hand_21x3):
    # TODO: 실제 각도 계산은 기존 전처리 코드와 맞게 구현
    if np.all(hand_21x3 == 0):
        return np.zeros(10, dtype=np.float32)
    return np.zeros(10, dtype=np.float32)

def calculate_hand_face_relation(lh_21x3, rh_21x3, face_468x3):
    if np.all(face_468x3 == 0):
        return np.zeros(6, dtype=np.float32)

    nose = face_468x3[1, :]  # face mesh 코 tip index
    lh_wrist = lh_21x3[0, :] if not np.all(lh_21x3 == 0) else nose
    rh_wrist = rh_21x3[0, :] if not np.all(rh_21x3 == 0) else nose

    vec_l = lh_wrist - nose
    vec_r = rh_wrist - nose

    dist_l = np.linalg.norm(vec_l)
    dist_r = np.linalg.norm(vec_r)

    feat = np.concatenate([vec_l, vec_r, [dist_l, dist_r]], axis=0)
    if feat.shape[0] > 6:
        feat = feat[:6]
    elif feat.shape[0] < 6:
        feat = np.pad(feat, (0, 6 - feat.shape[0]))
    return feat.astype(np.float32)

def extract_feature(results):
    pose_33x4, face_468x3, lh_21x3, rh_21x3 = landmarks_to_np(results)

    if USE_CANON:
        pose_c, face_c, lh_c, rh_c = canonicalize(pose_33x4, face_468x3, lh_21x3, rh_21x3)
    else:
        pose_c, face_c, lh_c, rh_c = pose_33x4[:, :3], face_468x3, lh_21x3, rh_21x3

    relative_lh, relative_rh = calculate_relative_hand_coords(lh_c, rh_c)
    angles_lh = calculate_finger_angles(lh_c)
    angles_rh = calculate_finger_angles(rh_c)
    rel_feat  = calculate_hand_face_relation(lh_c, rh_c, face_c)

    feat152 = np.concatenate([relative_lh, relative_rh, angles_lh, angles_rh, rel_feat]).astype(np.float32)
    return feat152, lh_c, rh_c


# =========================
# 5) 보정 / 등급 / 상태 전환
# =========================
def softmax_T(logits, T):
    return F.softmax(logits / T, dim=-1)

def exp_squash_scalar(p, beta):
    eb = np.exp(beta)
    num = np.exp(beta * p) - 1.0
    den = eb - 1.0 + 1e-12
    return float(np.clip(num / den, 0.0, 1.0))

def grade_from_prob(p):
    if p >= THR_PERFECT:
        return "Perfect"
    if p >= THR_OK:
        return "OK"
    if p >= THR_NOTBAD:
        return "Not Bad"
    return "Bad"

def draw_text(img, text, position, color=(255, 0, 255)):
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=FONT, fill=color)
    return np.array(img_pil)

def advance_to_next_word():
    """
    현재 문장에서 다음 단어로 이동.
    마지막 단어까지 끝나면 False 리턴 (all_done 처리용)
    """
    global current_word_idx, current_target_label, current_target_idx

    current_word_idx += 1
    if current_word_idx >= len(sentence_words):
        current_word_idx = len(sentence_words) - 1
        return False  # 문장 끝

    current_target_label, current_target_idx = get_current_target()
    return True

def reset_current_sentence():
    global current_word_idx, current_target_label, current_target_idx
    current_word_idx = 0
    current_target_label, current_target_idx = get_current_target()


# =========================
# 6) 실시간 루프 (s로 시작)
# =========================
cap = cv2.VideoCapture(0)
sequence    = deque(maxlen=FRAME_TARGET)
hit_history = deque(maxlen=HIT_WINDOW)

result_text   = "s를 누르면 퀴즈를 시작합니다."
frame_counter = 0
all_done      = False
quiz_active   = False  # s 누르기 전까지는 False

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(img_rgb)

        # ---- 상단 UI: 현재 문장/단어 진행 상황 ----
        sentence_str = " ".join(sentence_words)
        status_msg_1 = f"문장: {sentence_str}"
        status_msg_2 = f"단어 {current_word_idx+1}/{len(sentence_words)}: '{current_target_label}'"

        frame = draw_text(frame, status_msg_1, (30, 20))
        frame = draw_text(frame, status_msg_2, (30, 40))

        # ---- 안내/피드백 문구 ----
        if not quiz_active:
            frame = draw_text(frame, "s를 눌러 퀴즈를 시작하세요.", (30, 70))
        else:
            frame = draw_text(frame, result_text, (30, 70))

        # 하단 키 안내
        frame = draw_text(frame, "키: s=시작, r=문장 리셋, n=단어 스킵, q=종료", (30, 420))

        # ---- 퀴즈가 활성화된 상태에서만 feature 쌓고 예측 ----
        if quiz_active and (not all_done):
            feat, lh, rh = extract_feature(results)
            if np.any(lh) or np.any(rh):
                sequence.append(feat)

            if (len(sequence) >= FRAME_MIN) and (current_target_idx is not None):
                frame_counter += 1
                if frame_counter % PRED_INTERVAL == 0:
                    xs_array = np.array(sequence, dtype=np.float32)
                    length   = xs_array.shape[0]

                    xb = torch.tensor(xs_array, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                    L  = torch.tensor([length], dtype=torch.long, device=DEVICE)

                    with torch.no_grad():
                        logits = model(xb, L)              # [1, C]
                        qT = softmax_T(logits, T=T)        # [1, C]
                        qT_np = qT.squeeze(0).cpu().numpy()

                    p_target_raw = float(qT_np[current_target_idx])
                    p_target_p   = exp_squash_scalar(p_target_raw, BETA)  # 0~1
                    grade        = grade_from_prob(p_target_p)
                    score_str    = f"{p_target_p*100:.1f}%"

                    result_text = f"[{grade}] {current_target_label}: {score_str}"

                    # ---- 퀵드로우 패스 로직 ----
                    is_hit = (p_target_p >= PASS_THRESHOLD)
                    hit_history.append(is_hit)

                    if sum(hit_history) >= HIT_REQUIRED and is_hit:
                        # 현재 단어 통과!
                        hit_history.clear()
                        sequence.clear()
                        progressed = advance_to_next_word()
                        if progressed:
                            all_done = False
                            result_text = f"정답! 다음 단어 → '{current_target_label}'"
                        else:
                            result_text = "문장을 모두 마쳤어요! q를 눌러 종료하거나 r로 다시 시작하세요."
                            all_done = True

        cv2.imshow(f"Sign Quiz - {CATEGORY}", frame)

        # ---- 키 입력 처리 ----
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('s'):
            # 퀴즈 시작 (대기 → 진행, 항상 새 라운드로)
            quiz_active = True
            all_done    = False
            hit_history.clear()
            sequence.clear()
            reset_current_sentence()
            result_text = f"퀴즈 시작! 첫 단어: '{current_target_label}'"

        elif key == ord('r'):
            # 현재 문장 리셋 (퀴즈는 계속 진행 상태)
            if quiz_active:
                hit_history.clear()
                sequence.clear()
                reset_current_sentence()
                all_done    = False
                result_text = f"문장을 처음부터 다시 시작합니다. 첫 단어: '{current_target_label}'"

        elif key == ord('n'):
            # 현재 단어 스킵하고 다음 단어
            if quiz_active and (not all_done):
                hit_history.clear()
                sequence.clear()
                progressed = advance_to_next_word()
                if progressed:
                    all_done = False
                    result_text = f"다음 단어로 이동: '{current_target_label}'"
                else:
                    result_text = "문장을 모두 마쳤어요! q=종료, r=다시 시작"
                    all_done = True

finally:
    cap.release()
    cv2.destroyAllWindows()
