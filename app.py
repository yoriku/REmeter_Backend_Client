from flask import Flask, request, jsonify
from azure.cosmos import CosmosClient
from flask_cors import CORS
import os
from PIL import Image
from dotenv import load_dotenv
from utils_ER_estimate import ER_estimate, get_image_PIL
import boto3
import datetime
import json
import pytz

load_dotenv()

app = Flask(__name__)
CORS(app)  # CORSを有効にする（異なるオリジンからのリクエストを許可）

# Cosmos DB の接続情報を指定します
endpoint = os.getenv("ENDPOINT")
key = os.getenv("KEY")
database_name = "REmetor-Backend"
container_name = "Rooms"

# Cosmos DB に接続します
client = CosmosClient(endpoint, key)

# Cosmos DBのコンテナを取得します
container = client.get_database_client(database_name).get_container_client(
    container_name
)

# Rekognition API の接続情報を指定します
key_id = os.getenv("EMOTION_KEY")
access_key = os.getenv("EMOTION_ACCESS")

rekognition = boto3.client(
    "rekognition",
    aws_access_key_id=key_id,
    aws_secret_access_key=access_key,
    region_name="us-east-1",
)


@app.route("/", methods=["GET"])
def hello():
    return "Hello"

@app.route("/getRoomsList", methods=["GET"])
def get_rooms_list():
    query = "SELECT c.Roomname, c.Description, c.Id FROM c"  # 部屋のリストを取得するクエリ
    items = list(container.query_items(query, enable_cross_partition_query=True))  # クエリを実行してアイテムのリストを取得
    return {"items": items}  # アイテムのリストをレスポンスとして返す


@app.route("/test", methods=["POST"])
def getImage():
    print(request.files)  # リクエストから送信されたファイルを取得
    image_file = request.files["image"]  # リクエストから画像ファイルを取得
    print(image_file)  # 取得した画像ファイルの情報を表示
    
    # ER estimate
    img_ndarray, img_bytes = get_image_PIL(path=image_file)  # 画像ファイルをPIL Imageに変換
    emotion, reaction = ER_estimate(rekognition, img_ndarray, img_bytes)  # 画像の感情と反応を推定
    image = Image.open(image_file)  # 画像ファイルをPIL Imageとして開く
    image.save("img/test.jpg")  # 画像を指定のパスに保存
    print(emotion, reaction)
    return {"success":True}  # 成功したことを示すレスポンスを返す


@app.route("/postAction/<username>", methods=["POST"])
def post_action(username):
    image_file = request.files["image"]  # リクエストから画像ファイルを取得
    room_id = request.args.get("room_id")  # リクエストからroom_idを取得
    print(room_id)
    img_ndarray, img_bytes = get_image_PIL(path=image_file)  # 画像ファイルをPIL Imageに変換
    emotion, reaction = ER_estimate(rekognition, img_ndarray, img_bytes)  # 画像の感情と反応を推定
    print(emotion, reaction)
    time = datetime.datetime.now(pytz.timezone('Asia/Tokyo')).strftime('%Y/%m/%d/%H:%M:%S')   # 現在の日時を取得
    
    send_data = {"Username": username, "Emotion": emotion,"Reaction":reaction, "Time":time}  # 送信するデータの作成
    
    result = container.read_item(item=room_id, partition_key=room_id)  # Cosmos DBから指定されたroom_idのアイテムを取得
    
    img_ndarray, img_bytes = get_image_PIL(path=image_file)  # 画像ファイルを再度PIL Imageに変換
    
    if "ActionsList" in result:  # アイテムに"ActionList"が含まれているかチェック

        result["ActionsList"].append(send_data)  # "ActionList"にデータを追加

    else:
        result["ActionsList"] = [send_data]  # "ActionList"を作成し、データを追加
    # container.ReplaceItem(result)
    container.replace_item(room_id, result)
    return jsonify({"Emotion": emotion, "Reaction": reaction})  # 推定された感情と反応を返す

    

if __name__ == "__main__":
    # アプリケーションをデバッグモードで実行し、ポート番号を環境変数から取得（デフォルトは5000）
    app.run(debug=True, host="0.0.0.0", port = 5000,threaded=True)
