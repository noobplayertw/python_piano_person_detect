import cv2
import numpy as np
import time
import base64
import configparser
import os
from datetime import datetime
from ultralytics import YOLO
from loguru import logger
from TaipeiON_Model import send_notification

# ==================== 參數化設定 (Configuration) ====================
# 建議：若遠距離辨識不佳，可將 model_path 改為 yolo11m.pt
CONFIG = {
    "model_path": "./yolo12m.pt",
    "video_source": "rtsp://root:root@10.242.9.127/cam1/onvif-h264",
    # "video_source": "./R9 陽光大廳東側_2-17_10.242.9.12_20260430_151000_7275468.avi",
    "stay_threshold": 60,          # 停留門檻 (秒)
    "station": "R06",    # 車站名稱
    "stage": "Piano_Area",          # 監測區域名稱
    "target_class": 0,              # YOLO 類別 (0 通常是人)
    "line_color": (0, 255, 0),      # 線條顏色
    "point_color": (0, 0, 255),     # 點的顏色
    "roi_color": (255, 0, 0),       # 完成後的區域顏色
    "config_file": "roi_config.ini", # 座標儲存檔案路徑
    "width": 1280,                  # 輸出寬度
    "height": 720,                  # 輸出高度
    "log_file": "logs/{time:YYYYMMDD}.log",
    "alert_dir": "pictures",            # 告警圖片儲存目錄
    "conf": 0.5,                    # 提高門檻以減少雜訊干擾追蹤
    "imgsz": 960,                   # 降低推論解析度以大幅提升 FPS，若辨識率下降再調回 1280
    "vlm_verification": False,      # 是否開啟 VLM 二次驗證 (需額外算力)
}

# ==================== 全域變數 ====================
points = []               # 儲存點
roi_polygon = None        # 最終多邊形
track_timers = {}         # 紀錄 ID 進入時間 {track_id: start_time}
notified_ids = set()      # 紀錄已發送過的 ID，避免重複發送

def save_roi_to_ini(pts):
    """將座標點儲存至 ini 檔"""
    config = configparser.ConfigParser()
    config['ROI'] = {
        'points': str(pts)
    }
    with open(CONFIG["config_file"], 'w') as configfile:
        config.write(configfile)
    print(f"區域座標已儲存至 {CONFIG['config_file']}")

def load_roi_from_ini():
    """從 ini 檔讀取座標點"""
    global points, roi_polygon
    if os.path.exists(CONFIG["config_file"]):
        try:
            config = configparser.ConfigParser()
            config.read(CONFIG["config_file"])
            if 'ROI' in config and 'points' in config['ROI']:
                pts_str = config['ROI']['points']
                # 使用 eval 將字串轉回 list of tuples
                points = eval(pts_str)
                if len(points) >= 3:
                    roi_polygon = np.array(points, np.int32)
                    print(f"已從檔案載入區域座標: {points}")
        except Exception as e:
            print(f"載入設定檔失敗: {e}")

def mouse_callback(event, x, y, flags, param):
    global points, roi_polygon, track_timers, notified_ids
    
    # 左鍵：點選點
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"點擊位置: ({x}, {y})")
        
    # 右鍵：連成線並完成區域
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(points) >= 3:
            roi_polygon = np.array(points, np.int32)
            save_roi_to_ini(points)
            print("區域設定完成並已儲存！")
        else:
            print("請至少點選 3 個點來組成區域")
            
    # 中鍵：重新設定區域
    elif event == cv2.EVENT_MBUTTONDOWN:
        points = []
        roi_polygon = None
        track_timers = {}
        notified_ids = set()
        if os.path.exists(CONFIG["config_file"]):
            os.remove(CONFIG["config_file"])
        print("區域已重置")

def vlm_verify_person(frame, box):
    """
    VLM 驗證預留接口。
    可以使用 local 的 Moondream 或 Ollama 模型進行確認。
    """
    if not CONFIG["vlm_verification"]:
        return True
    
    # 這裡可以實作呼叫 VLM 的邏輯，例如：
    # "Is there a real person in this cropped image?"
    # 目前回傳 True 代表跳過驗證
    return True

def get_base64_image(img):
    """將影像轉成 Base64 字串"""
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')

def main():
    global points, roi_polygon, track_timers

    # 0. 確保目錄存在並設定 Log
    if not os.path.exists(CONFIG["alert_dir"]):
        os.makedirs(CONFIG["alert_dir"])
    log_dir = os.path.dirname(CONFIG["log_file"])
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger.add(CONFIG["log_file"], rotation="00:00", retention="30 days", level="INFO")

    # 0. 啟動時載入現有區域設定
    load_roi_from_ini()

    # 1. 載入模型
    model = YOLO(CONFIG["model_path"])
    
    # 2. 開啟 RTSP 串流
    cap = cv2.VideoCapture(CONFIG["video_source"])
    # 優化：減少 OpenCV 內部緩存，減少延遲
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print(f"無法開啟串流: {CONFIG['video_source']}")
        return

    window_name = "YOLOv12 Person Stay Monitor"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret:
            # 若是跳畫面太嚴重，有些情況需要跳過壞幀
            continue
            
        # 技巧：如果發現嚴重延遲，可以手動抓取最新幀（丟棄緩存）
        # for _ in range(5): cap.grab() 

            print("無法讀取影像，嘗試重新連接...")
            cap = cv2.VideoCapture(CONFIG["video_source"])
            continue

        # 強制調整畫面大小為 1280*720
        frame = cv2.resize(frame, (CONFIG["width"], CONFIG["height"]))

        # 3. 執行追蹤 (改為全畫面偵測，以獲得更穩定的 Tracker ID)
        results = model.track(
            frame, 
            persist=True, 
            classes=[CONFIG["target_class"]], 
            conf=CONFIG["conf"], 
            imgsz=CONFIG["imgsz"],
            tracker="bytetrack.yaml", 
            verbose=False,
            device=0
        )
        
        # 4. 繪製當前編輯中的線段
        for pt in points:
            cv2.circle(frame, pt, 4, CONFIG["point_color"], -1)
        if len(points) > 1:
            for i in range(len(points) - 1):
                cv2.line(frame, points[i], points[i+1], CONFIG["line_color"], 2)

        # 5. 如果 ROI 已設定，開始判斷
        if roi_polygon is not None:
            cv2.polylines(frame, [roi_polygon], True, CONFIG["roi_color"], 3)
            
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                
                for box, tid in zip(boxes, track_ids):
                    # box 格式為 [x1, y1, x2, y2]
                    # 計算中心點 (中心偏底部的點較準確)
                    cx = int((box[0] + box[2]) / 2)
                    cy = int((box[1] + box[3]) / 2)
                    
                    # 判斷點是否在區域內
                    # cv2.pointPolygonTest 回傳值 >= 0 代表在內部或邊緣
                    is_inside = cv2.pointPolygonTest(roi_polygon, (cx, cy), False) >= 0
                    
                    if is_inside:
                        # 進入區域計時
                        if tid not in track_timers:
                            track_timers[tid] = time.time()
                        
                        stay_duration = time.time() - track_timers[tid]
                        
                        # 顯示停留秒數
                        cv2.putText(frame, f"ID:{tid} Stay:{int(stay_duration)}s", 
                                    (int(box[0]), int(box[1]-10)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)  # 這裡改為綠色 (0, 255, 0)
                        
                        # 超過門檻且該 ID 尚未成功發送過通知
                        if stay_duration > CONFIG["stay_threshold"] and tid not in notified_ids:
                            # VLM 二次確認 (防止遠處雜訊誤判)
                            if vlm_verify_person(frame, box):
                                print(f"觸發報警！ID: {tid} 停留超時")
                            
                                # 準備通知資料
                                ts_now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f0+08:00")
                                file_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                                b64_img = get_base64_image(frame)
                            
                                # 1. 儲存本地告警圖片
                                img_filename = f"alert_ID{tid}_{file_ts}.jpg"
                                img_path = os.path.join(CONFIG["alert_dir"], img_filename)
                                cv2.imwrite(img_path, frame)
                            
                                # 2. 紀錄 Log
                                logger.info(f"告警觸發: ID {tid} 在 {CONFIG['stage']} 停留達 {int(stay_duration)}秒, 圖片已儲存: {img_path}")
                            
                                # 發送通知
                                status, resp = send_notification(
                                    timestamp=ts_now,
                                    station=CONFIG["station"],
                                    stage=CONFIG["stage"],
                                    index=tid,
                                    pic_base64=b64_img,
                                    alert_type="fail"
                                )
                            
                                if status in [200, 201, 500]:
                                    notified_ids.add(tid)
                    else:
                        # 離開區域則移除計時
                        if tid in track_timers:
                            del track_timers[tid]

        # 6. 顯示影像
        cv2.imshow(window_name, frame)
        
        # 按 'q' 退出，'c' 手動清除 ROI
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            points = []
            roi_polygon = None
            track_timers = {}

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
