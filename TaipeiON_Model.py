import httpx
from loguru import logger
from datetime import datetime
import base64
import cv2

def send_notification(timestamp, station, stage, index, pic_base64, alert_type="fail"):
    """
    發送符合QtRTSP.py格式的HTTP通知
    """
    try:
        # 解析時間戳以獲取DAY和DAYTIME
        # 假設timestamp格式為"2025-08-19T15:20:19.3500000+08:00"
        dt = datetime.strptime(timestamp.split('.')[0], "%Y-%m-%dT%H:%M:%S")
        DAY = dt.strftime("%Y%m%d")
        DAYTIME = dt.strftime("%Y-%m-%dT%H:%M:%S")

        # 根據告警類型設定SUB_SYSTEM_TYPE
        if alert_type == "fail":
            # sub_system_type = "滯留物及跌倒偵測"
            # sub_system_type = "北捷創新課告警訊息"
            sub_system_type = "鋼琴使用者偵測"
            alert_memo = f"{station}{stage}偵測到人員在鋼琴區域。"
        else:  # normal
            # sub_system_type = "滯留物及跌倒偵測"
            # sub_system_type = "北捷創新課告警訊息"
            sub_system_type = "鋼琴使用者偵測"
            alert_memo = f"{station}{stage}偵測到人員在鋼琴區域。"

        # 建構通知資料（符合QtRTSP.py中的格式）
        notification_data = {
            "DAY": DAY,
            "DAYTIME": DAYTIME,
            "ALERT_MEMO": alert_memo,
            "LINE": "R",
            "LOCATION": station,
            "SUB_SYSTEM_TYPE": sub_system_type,
            "Image_Data_Base64": pic_base64,
            "Flag": "Y"
        }

        logger.info("發送資料預覽:")
        logger.info(f"  DAY: {notification_data['DAY']}")
        logger.info(f"  DAYTIME: {notification_data['DAYTIME']}")
        logger.info(f"  ALERT_MEMO: {notification_data['ALERT_MEMO']}")
        logger.info(f"  LINE: {notification_data['LINE']}")
        logger.info(f"  LOCATION: {notification_data['LOCATION']}")
        logger.info(f"  SUB_SYSTEM_TYPE: {notification_data['SUB_SYSTEM_TYPE']}")
        logger.info(
            f"  Image_Data_Base64 長度: {len(notification_data['Image_Data_Base64']) if notification_data['Image_Data_Base64'] else 0}")
        logger.info(f"  Flag: {notification_data['Flag']}")

        # 發送POST請求
        url = "http://10.36.3.111/NotifyStorageAPI/api/NotifyStorage/UploadImageAndNotify" #即時發送(需要的話要通知LSJ修改)
        # url = "http://10.36.3.111//NotifyStorageAPI/api/NotifyStorage/UploadImage" # 非即時發送(需要的話要通知LSJ修改約1分鐘延遲)
        with httpx.Client(verify=False) as client:
            response = client.post(url, json=notification_data)
            logger.info(f"伺服器回應狀態: {response.status_code}")
            logger.info(f"伺服器回應內容: {response.text}")

            # 記錄詳細的回應資訊
            logger.info(f"回應標頭資訊:")
            for key, value in response.headers.items():
                logger.info(f"  {key}: {value}")

            # 根據回應狀態碼記錄相應的資訊
            if response.status_code == 200:
                logger.success("告警訊息發送成功")
            elif response.status_code == 201:
                logger.success("告警訊息建立成功")
            elif response.status_code == 400:
                logger.error("請求參數錯誤，請檢查發送的資料格式")
            elif response.status_code == 401:
                logger.error("身分驗證失敗")
            elif response.status_code == 403:
                logger.error("存取被拒絕")
            elif response.status_code == 404:
                logger.error("API端點未找到")
            elif response.status_code == 500:
                logger.error("伺服器內部錯誤")
            else:
                logger.warning(f"未知的回應狀態: {response.status_code}")

            return response.status_code, response.text

    except httpx.RequestError as e:
        logger.error(f"發送請求時發生網路錯誤: {e}")
        return None, str(e)
    except Exception as e:
        logger.error(f"發送通知時出錯: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, str(e)
