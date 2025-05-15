import os
from flask import Flask, jsonify, request
import logging
import json
import tempfile
import mimetypes
import datetime
from typing import List
from flask_cors import CORS
# 导入 Vertex AI 相关库
from google import genai
from google.genai import types
from google.cloud.exceptions import NotFound
from google.api_core.exceptions import GoogleAPICallError, ResourceExhausted
import vertexai
from vertexai.vision_models import Image, MultiModalEmbeddingModel
from google.cloud import firestore
from google.cloud import aiplatform
from google.cloud.aiplatform_v1.types import IndexDatapoint

# --- 配置 ---
GCP_PROJECT = "baidao-test-666808"        # 固定项目 ID
PRIMARY_REGION = "global"                 # 主区域
FALLBACK_REGION = "us-central1"           # 备用区域
GEMINI_MODEL_NAME = "gemini-2.0-flash-lite-001"  # 固定模型名称
DATABASE_NAME = "image-info"             # Firestore 数据库名称
COLLECTION_NAME = "image-info"           # Firestore 集合名称
INDEX_NAME = "projects/746866758104/locations/us-central1/indexes/2416253767851704320" # 固定索引名称
INDEX_ENDPOINT_NAME = "projects/746866758104/locations/us-central1/indexEndpoints/1344038615746871296" # 固定索引端点名称
DEPLOYED_INDEX_ID = "image_vector_search_test" # 固定部署索引 ID
NUM_NEIGHBORS = 100 # 向量搜索返回的邻居数量
THRESHOLD = 0.07  # 匹配阈值
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}  # 允许的文件扩展名
MAX_IMAGE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB
# --- 固定的提示词 (保持不变) ---
FIXED_PROMPT = """You are a professional image analyst. You will analyze the input image strictly according to the following steps and output the results in a structured format without additional explanations.
Step 1: Object Detection and Object Attribute Recognition
Detect objects in the image that match the following 5 categories and record the color, shape, and description of the objects.
1. "person": The human body or something with typical human characteristics, such as limbs, torso, human postures, etc.
2. "human face": Close - ups or parts of the human face, which must have clear human features (such as facial features, expressions, skin texture, etc.).
3. "vehicle": Vehicles with clear transportation vehicle characteristics (such as wheels, engines, carriage structures, etc.), ignoring toy cars.
4. "animal": Non - human biological individuals, including wild animals, domestic pets, farm animals, etc., which must have animal characteristics (such as fur, scales, feathers, etc.).
5. "package": Containers or parcels used to enclose items, which must have clear packaging characteristics (such as box - shaped, bag - shaped, sealed packaging, etc.).
Step 2: Keyword Extraction
Based on the entire image, output 3 - 5 keywords separated by commas: (Keywords: XXX, XXX, XXX)
Step 3: Overall Description
Generate a short description of the entire picture."""

# 初始化 Flask 应用
app = Flask(__name__)
CORS(app) # 允许跨域请求
app.config['MAX_CONTENT_LENGTH'] = MAX_IMAGE_SIZE_BYTES  # Flask 内置限制上传大小

logging.basicConfig(level=logging.INFO)

# --- 全局 Firestore 和 Vertex AI Client配置 ---
try:
    firestore_db = firestore.Client(project=GCP_PROJECT, database=DATABASE_NAME)
    logging.info("启动时初始化 Firestore 客户端成功")
except Exception as e:
    logging.error("启动时初始化 Firestore 客户端失败")
    firestore_db = None
    raise RuntimeError(f"启动时初始化 Firestore 客户端时发生内部错误: {e}") from e

try:
    vertexai.init(project=GCP_PROJECT, location=FALLBACK_REGION)
    mm_model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
    logging.info("启动时Vertex AI和多模态模型初始化成功")
except Exception as e:
    logging.error(f"启动时初始化多模态嵌入模型时出错: {e}")
    raise RuntimeError("启动时应用初始化失败，请检查 GCP 配置") from e

try:
    # 初始化 Vertex AI 客户端
    aiplatform.init(project=GCP_PROJECT, location=FALLBACK_REGION)

    # Create the index endpoint instance from an existing endpoint.
    my_index_endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name=INDEX_ENDPOINT_NAME)
except Exception as e:
    logging.error(f"启动时初始化Vertex AI IndexEndpoint时出错: {e}")
    raise RuntimeError("启动时应用初始化失败，请检查 GCP 配置") from e

# --- 生成配置 (根据需要调整) ---
generate_content_config = types.GenerateContentConfig(
    temperature = 1,
    top_p = 0.95,
    max_output_tokens = 8192,

    safety_settings = [types.SafetySetting(
        category="HARM_CATEGORY_HATE_SPEECH",
        threshold="OFF"
    ),types.SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT",
        threshold="OFF"
    ),types.SafetySetting(
        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
        threshold="OFF"
    ),types.SafetySetting(
        category="HARM_CATEGORY_HARASSMENT",
        threshold="OFF"
    )],
    response_mime_type="application/json",
    response_schema=genai.types.Schema(
        type = genai.types.Type.OBJECT,
        required = ["objects", "summary", "feature"],
        properties = {
            "objects": genai.types.Schema(
                type = genai.types.Type.ARRAY,
                items = genai.types.Schema(
                    type = genai.types.Type.OBJECT,
                    required = ["category", "attributes", "action", "summary"],
                    properties = {
                        "category": genai.types.Schema(
                            type = genai.types.Type.STRING,
                            enum = ["person", "face", "vehicle", "animal", "package"],
                        ),
                        "attributes": genai.types.Schema(
                            type = genai.types.Type.OBJECT,
                            required = ["color", "shape", "clothing"],
                            properties = {
                                "color": genai.types.Schema(
                                    type = genai.types.Type.STRING,
                                    nullable = "True",
                                ),
                                "shape": genai.types.Schema(
                                    type = genai.types.Type.STRING,
                                    nullable = "True",
                                ),
                                "clothing": genai.types.Schema(
                                    type = genai.types.Type.STRING,
                                    nullable = "True",
                                ),
                            },
                        ),
                        "action": genai.types.Schema(
                            type = genai.types.Type.STRING,
                            description = "Describes the object's action or behavior, e.g., walking, running, staying, carrying",
                        ),
                        "summary": genai.types.Schema(
                            type = genai.types.Type.STRING,
                            description = "A short summary describing the object",
                        ),
                    },
                ),
            ),
            "summary": genai.types.Schema(
                type = genai.types.Type.STRING,
                description = "Short summary describing the whole picture.",
            ),
            "feature": genai.types.Schema(
                type = genai.types.Type.STRING,
                description = "Extract and aggregate key descriptive keywords from the summary of each object to represent the main visual features in the image.",
            ),
        },
    ),
)

# --- 允许的文件扩展名检查 ---
def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
# --- MIME 类型映射 (根据需要扩展) ---
def get_mime_type_from_url(gcs_url: str) -> str:
    """根据 GCS url 的文件后缀推断 MIME 类型"""
    # 尝试使用 mimetypes 库
    mime_type, _ = mimetypes.guess_type(gcs_url)
    if mime_type:
        return mime_type

    # 如果 mimetypes 失败，使用简单的后缀映射作为后备
    filename = gcs_url.split('/')[-1].lower()
    if filename.endswith('.png'):
        return 'image/png'
    elif filename.endswith('.jpg'):
        return 'image/jpg'
    elif filename.endswith('.jpeg'):
        return 'image/jpeg'
    else:
        logging.warning(f"无法从 url {gcs_url} 推断 MIME 类型，将返回 None。")
        return None

# --- Gemini Vision 图像分析函数 ---
def analyze_image_with_gemini_fallback(
    gcs_url: str,
    primary_region: str = PRIMARY_REGION,
    fallback_region: str = FALLBACK_REGION,
    model_name: str = GEMINI_MODEL_NAME,
    prompt: str = FIXED_PROMPT,
) ->  str:
    """
    使用 Gemini Vision 模型分析 GCS 中的图像，并实现区域回退逻辑。

    Args:
        gcs_url: 图像在 Google Cloud Storage 中的 url (例如 "gs://bucket-name/image.jpg").
        project_id: Google Cloud 项目 ID.
        primary_region: 首选的 Vertex AI 区域.
        fallback_region: 当首选区域资源耗尽 (429) 时尝试的备用区域.
        model_name: 要使用的 Gemini 模型名称 (例如 "gemini-1.0-pro-vision").
        prompt: 提供给模型的文本提示.
        generation_config: Gemini 模型的生成配置.

    Returns:
        如果 Gemini 返回有效的 JSON 字符串，则返回解析后的 Python 字典。
        否则，返回 Gemini 返回的原始文本字符串。
        同时返回使用的区域和原始响应对象（用于调试）。

    Raises:
        ValueError: 如果无法从 GCS url 推断 MIME 类型或输入无效。
        FileNotFoundError: 如果 GCS url 指向的文件不存在或无权访问。
        RuntimeError: 如果在准备请求或处理期间发生意外错误。
        google.api_core.exceptions.ResourceExhausted: 如果主区域和备用区域均资源耗尽。
        google.api_core.exceptions.GoogleAPICallError: 如果发生其他 Google API 调用错误。
        Exception: 其他未预料到的底层错误。
    """
    mime_type = get_mime_type_from_url(gcs_url)
    if not mime_type:
        raise ValueError(f"无法从 GCS url '{gcs_url}' 的后缀推断出支持的 MIME 类型")

    logging.info(f"开始分析图片: {gcs_url}, MIME 类型: {mime_type}")

    # 准备 Gemini 请求内容
    try:
        text_part = types.Part.from_text(text=prompt)
        image_part = types.Part.from_url(file_url=gcs_url, mime_type=mime_type)
    except NotFound as e:
        logging.error(f"GCS 文件未找到或无权访问: {gcs_url}", exc_info=True)
        raise FileNotFoundError(f"无法访问提供的 GCS url: {gcs_url} (文件不存在或无权限)") from e
    except Exception as e:
        logging.error(f"准备 Gemini 输入时出错 (非 GCS 访问问题): {e}", exc_info=True)
        raise RuntimeError(f"准备 API 请求时内部错误: {e}") from e

    current_region = PRIMARY_REGION
    model = None
    response = None
    last_error = None

    contents = [
        types.Content(
        role="user",
        parts=[
            text_part,
            image_part
        ]
        ),
    ]

    for attempt_region in [PRIMARY_REGION, FALLBACK_REGION]:
        current_region = attempt_region
        try:
            gemini_client = genai.Client(
            vertexai=True,
            project=GCP_PROJECT,
            location=current_region,
            )
            # 初始化模型
            model = GEMINI_MODEL_NAME

            logging.info(f"尝试在区域 '{current_region}' 调用 Gemini 模型 '{model_name}'...")

            # --- 调用 Gemini API ---
            # 注意：新版 SDK 的调用方式可能略有不同，请参考最新文档
            response = gemini_client.models.generate_content(
                model = model,
                contents = contents,
                config = generate_content_config,
            )
            # --- 结束调用 ---

            logging.info(f"在区域 '{current_region}' 成功收到 Gemini 响应。")
            region_used = current_region
            # 如果成功，跳出循环
            break

        except ResourceExhausted as e: # 特别捕获 429 错误
            last_error = e
            logging.warning(f"在区域 '{current_region}' 遇到资源耗尽 (429) 错误: {e}")
            if current_region == fallback_region:
                logging.error(f"主区域 '{primary_region}' 和备用区域 '{fallback_region}' 均遇到 429 错误，停止尝试。")
                # 不在此处引发，让循环结束后的检查来处理
            else:
                logging.info(f"准备尝试备用区域 '{fallback_region}'...")
                # 继续循环到下一个区域

        except GoogleAPICallError as e:
            last_error = e
            logging.error(f"在区域 '{current_region}' 调用 Vertex AI API 时出错 (非 429): {e}", exc_info=True)
            # 对于非 429 的 API 错误，通常不应重试，直接抛出异常
            raise # 重新引发捕获的异常，保留原始类型和信息

        except Exception as e:
            last_error = e
            logging.error(f"在区域 '{current_region}' 处理图片分析时发生未知错误: {e}", exc_info=True)
            # 对于未知错误，也直接抛出
            raise RuntimeError(f"服务器内部未知错误 ({current_region}): {e}") from e

    # 循环结束后，检查 response 是否成功获取
    if response is None or region_used is None:
        # 如果 response 为 None，说明所有尝试都失败了
        logging.error(f"所有区域尝试均失败，最后错误: {last_error}")
        error_message = f"所有区域 ({primary_region}, {fallback_region}) 的 Gemini API 调用均失败。"
        if isinstance(last_error, ResourceExhausted):
             # 特别是 429 错误，明确指出
            error_message = f"Gemini API 资源耗尽 (429)，主区域 '{primary_region}' 和备用区域 '{fallback_region}' 均失败。"
            raise ResourceExhausted(f"{error_message} Last error: {last_error}") from last_error
        elif isinstance(last_error, GoogleAPICallError):
            raise GoogleAPICallError(f"{error_message} Last error: {last_error}") from last_error
        elif last_error:
             raise RuntimeError(f"{error_message} Last error: {last_error}") from last_error
        else:
             # 应该不会到这里，但作为保险
             raise RuntimeError(error_message + " 未记录具体错误。")


    # 处理成功的响应
    try:
        # 检查是否有有效内容返回 (考虑安全过滤等情况)
        if not response.candidates:
             logging.warning(f"Gemini 响应 (区域: {region_used}) 没有 candidate。响应详情: {response}")
             # 可以选择返回特定值或引发异常
             raise ValueError(f"Gemini 没有返回有效的分析结果 (无 candidate). Feedback: {response.prompt_feedback}")

        # 假设我们总是取第一个 candidate
        candidate = response.candidates[0]

        # 检查是否有文本部分
        if not candidate.content or not candidate.content.parts or not candidate.content.parts[0].text:
             logging.warning(f"Gemini 响应 (区域: {region_used}) 的 candidate 中没有文本内容。响应详情: {response}")
             # 根据需要处理，可能因为安全设置或其他原因被阻止
             # finish_reason 可能提供线索: candidate.finish_reason
             raise ValueError(f"Gemini 返回的 candidate 中无文本内容 (Finish Reason: {candidate.finish_reason}). Feedback: {response.prompt_feedback}")

        analysis_text = candidate.content.parts[0].text
        logging.info(f"成功提取 Gemini 分析文本 (来自区域: {region_used})。")

        # 尝试解析 JSON
        try:
            analysis_json = json.loads(analysis_text)
            logging.info("Gemini 输出成功解析为 JSON。")
            # 返回包含结果、区域和原始响应的字典
            return analysis_json
        except json.JSONDecodeError:
            logging.warning("Gemini 输出不是有效的 JSON 格式，将按原样返回文本。")
            # 返回包含结果、区域和原始响应的字典
            return {
                "analysis": analysis_text, # 返回原始文本
                "region_used": region_used,
                "raw_response": response
            }

    except Exception as e:
         # 捕获处理响应时可能出现的任何其他错误
         logging.error(f"处理 Gemini 响应时出错: {e}", exc_info=True)
         raise RuntimeError(f"处理 Gemini 响应时发生内部错误: {e}") from e

# --- image多模态嵌入函数 ---
def image_multimodalembedding(image_path: str, model: str) -> str:

    try:
        image = Image.load_from_file(image_path)
        embeddings = model.get_embeddings(
            image=image,
            # contextual_text="Colosseum",文本
            dimension=1408,
        )
        logging.info(f"Image Embedding 成功!")
        return embeddings.image_embedding
    except Exception as e:
        logging.error(f"处理图片嵌入时出错: {e}", exc_info=True)
        raise RuntimeError(f"处理图片嵌入时发生内部错误: {e}") from e

# --- text多模态嵌入函数 ---
def text_multimodalembedding(text: str, model: str):
    try:
        embeddings = model.get_embeddings(
            contextual_text=text,
            dimension=1408,
        )
        logging.info(f"Text Embedding 成功!")
        return embeddings.text_embedding
    except Exception as e:
        logging.error(f"处理图片嵌入时出错: {e}", exc_info=True)
        raise RuntimeError(f"处理图片嵌入时发生内部错误: {e}") from e

# --- 向量搜索最相邻 ---
def vector_search_find_neighbors(
    queries: List[List[float]],
    num_neighbors: int,
) -> None:
    """Query the vector search index.

    Args:
        project (str): Required. Project ID
        location (str): Required. The region name
        index_endpoint_name (str): Required. Index endpoint to run the query
        against.
        deployed_index_id (str): Required. The ID of the DeployedIndex to run
        the queries against.
        queries (List[List[float]]): Required. A list of queries. Each query is
        a list of floats, representing a single embedding.
        num_neighbors (int): Required. The number of neighbors to return.
    """
    logging.info(f"开始向量搜索 (区域: {FALLBACK_REGION})!")
    try:
        # 查询索引端点以获取最近邻
        resp_neighbors = my_index_endpoint.find_neighbors(
            deployed_index_id=DEPLOYED_INDEX_ID,
            queries=[queries],
            num_neighbors=num_neighbors,
        )
        logging.info(f"成功查询索引端点 (区域: {FALLBACK_REGION})。")
        logging.info(f"最近邻结果: {resp_neighbors}")
        return resp_neighbors

    except Exception as e:
        logging.error(f"向量计算时出错: {e}", exc_info=True)
        raise RuntimeError(f"向量计算时发生内部错误: {e}") from e
    
# --- 向量搜索最相邻排序 ---
def convert_match_neighbors_to_list_of_dicts(match_neighbor_data):
    """
    将包含MatchNeighbor对象列表的列表转换为列表的字典，其中每个字典代表一个邻居。

    这种格式可以更容易进行后续过滤或排序“距离”等属性。

    Arguments：
        match_neighbor_data：包含单个内部列表的列表，其中
        内部列表包含MatchNeighbor对象。
        示例格式：[[MatchNeighbor（…）， MatchNeighbor（…）]

    return：
        字典列表，其中每个字典都包含“id”，
        MatchNeighbor的'距离'和'crowding_tag'。返回一个空
        列出输入是否无效或为空。

    示例输出：
        [
            {'id'：'…'，'距离'：…，'crowding_tag'：'…'}，
            {'id'：'…'，'距离'：…，'crowding_tag'：'…'}
        ]
    """
    # 对预期的 [[...]] 结构进行基本验证
    if not match_neighbor_data or not isinstance(match_neighbor_data, list) or not match_neighbor_data[0] or not isinstance(match_neighbor_data[0], list):
        logging.warning("Warning: Input data format is not as expected (expected [[MatchNeighbor,...]]). Returning empty list.")
        # print("警告：输入数据格式与预期不符（预期为 [[MatchNeighbor,...]]）。返回空列表。")
        raise ValueError("Invalid input data format")

    inner_list = match_neighbor_data[0]
    result_list = []

    for neighbor in inner_list:
        # 检查该项目是否具有所需的属性
        if hasattr(neighbor, 'id') and hasattr(neighbor, 'distance') and hasattr(neighbor, 'crowding_tag'):
            neighbor_info = {
                'id': neighbor.id,
                'distance': neighbor.distance,
                'crowding_tag': neighbor.crowding_tag
                # 如果需要，在此处添加其他所需的属性
            }
            result_list.append(neighbor_info)
        else:
            logging.warning(f"Warning: Skipping item {neighbor} as it does not have expected attributes (id, distance, crowding_tag).")
            raise ValueError("Item does not have expected attributes (id, distance, crowding_tag)")
    logging.info(f"最近邻结果: {result_list}")

    # 过滤掉距离小于阈值的邻居
    logging.info(f"过滤掉距离小于阈值的邻居,当前阈值: {THRESHOLD}")
    filtered_neighbors = [
        neighbor for neighbor in result_list if neighbor['distance'] > THRESHOLD
    ]

    return filtered_neighbors

# --- 唯一ID函数 ---
def get_unique_id():
    # 1. 获取当前时间 
    utc_plus_8 = datetime.timezone(datetime.timedelta(hours=8), name='Asia/Shanghai') # name 是可选的，但有助于理解
    # 2. 获取当前的 UTC+8 时间
    # datetime.now() 可以接受一个 tz 参数来获取特定时区的当前时间
    now_utc8 = datetime.datetime.now(tz=utc_plus_8)
    # 3. 格式化时间为 "YYYYMMDDHHMMSS"
    time_prefix = now_utc8.strftime("%Y%m%d%H%M%S")
    # 4. 获取当前时间的微秒部分 (0-999999)
    microseconds = now_utc8.microsecond
    # 将时间前缀和6位微秒数组合在一起，微秒部分用0补足确保总是6位数
    unique_id = f"{time_prefix}{microseconds:06d}"

    return unique_id

# --- 插入数据到Firestore&Vector Search ---
def insert_info_embdding_data(json_data, unique_id, data_point):
    """插入Firestore
    Args:
        unique_id (str): 唯一ID
        vector_datapoint (Dict[str, Any]): 向量数据点
        bucket_name (str): GCS存储桶名称
    Returns:
        str: Firestore文档ID
    """
    try:
        # 1. 初始化 Firestore 客户端
        db = firestore.Client(project=GCP_PROJECT, database=DATABASE_NAME)
        collection_ref = db.collection(COLLECTION_NAME)
        # 获取文档引用并写入数据
        # 我们使用提供的 unique_id 作为 Firestore 文档的 ID。
        # 这使得以后可以轻松地按此 ID 查找或更新此特定文档。
        doc_ref = collection_ref.document(unique_id)
        # 使用 set() 方法写入数据。
        # - 如果具有该 ID 的文档不存在，它将创建新文档。
        # - 如果具有该 ID 的文档已存在，它将完全覆盖现有文档。
        #   (如果只想更新部分字段或确保只在不存在时创建，可以使用 merge=True 或 create())
        doc_ref.set(json_data)
        logging.info(f"Successfully wrote data with ID '{unique_id}' to collection '{COLLECTION_NAME}'.")
        # --- 可选：如果你想让 Firestore 自动生成文档 ID ---
        # 如果你不关心文档 ID 或没有自然的唯一标识符，可以使用 add()
        # auto_id_doc_ref, write_result = collection_ref.add(json_data)
        # print(f"Successfully wrote data with auto-generated ID '{auto_id_doc_ref.id}' to collection '{COLLECTION_NAME}'.")
        # 注意：如果使用 add()，那么 json_data 中的 "id" 字段需要单独处理，因为它不再是文档本身的 ID。
    except Exception as e:
        logging.error(f"Firestore数据入库时出错: {e}")
        raise RuntimeError(f"数据入库时发生内部错误: {e}") from e
    
    try:
        # 2. 初始化 Vector Search 客户端
        aiplatform.init(project=GCP_PROJECT, location=FALLBACK_REGION)
        index = aiplatform.MatchingEngineIndex(index_name=INDEX_NAME)
        index.upsert_datapoints(datapoints=data_point)
        logging.info(f"Successfully wrote data to index '{INDEX_NAME}'.")

        return True, "成功写入数据到Firestore和Vector Search!"

    except Exception as e:
        logging.error(f"Vector数据入库时出错: {e}")
        doc_ref.delete()
        raise RuntimeError(f"数据入库时发生内部错误: {e}") from e

# --- firestore数据检索 ---
def get_valid_gcs_url_items(data_list, firestore_client, collection_name):
    """
    从Firestore中检索列表中ID对应的gcs_url，并仅返回那些成功找到
    非空gcs_url的项（包含原始信息和gcs_url）。

    Args:
        data_list (list): 一个字典列表，每个字典可能包含一个'id'键以及其他字段。
                          例如: [{'id': 'id1', 'distance': 1.0, ...}, {'id': 'id2', ...}]
                          列表可能为空或包含非字典元素（会被忽略）。
        firestore_client (google.cloud.firestore.Client): 已初始化的Firestore客户端实例。
        collection_name (str): 要查询的Firestore集合的名称。

    Returns:
        list: 一个新的字典列表。只包含输入列表中那些对应ID在Firestore中存在、
              且其文档包含一个非空 'gcs_url' 字段的项。
              每个输出字典是原始字典的副本，并添加了从Firestore获取的 'gcs_url'。
              查询失败、未找到文档、或文档缺少 'gcs_url' 字段的项将被忽略。
    """  
    output_list = [] # 初始化一个空列表用于存储结果
    if not isinstance(data_list, list):
        print("警告：输入不是一个列表。")
        return output_list # 返回空列表
    if not firestore_client or not collection_name:
        print("错误：未提供 Firestore 客户端或集合名称。")
        return output_list # 返回空列表

    collection_ref = firestore_client.collection(collection_name)

    logging.info(f"\n开始从集合 '{collection_name}' 检索 GCS URLs 并筛选结果...")

    for item in data_list:
        # 检查item是否为字典，并安全地获取id
        if isinstance(item, dict):
            item_id = item.get('id')

            # 确保id是有效的非空字符串
            if isinstance(item_id, str) and item_id:
                gcs_url = None # 初始化 gcs_url 为 None
                try:
                    doc_ref = collection_ref.document(item_id)
                    doc_snapshot = doc_ref.get()

                    if doc_snapshot.exists:
                        doc_data = doc_snapshot.to_dict()
                        # 安全地获取gcs_url
                        retrieved_url = doc_data.get('gcs_url')
                        # 检查获取到的URL是否为非空字符串
                        if isinstance(retrieved_url, str) and retrieved_url:
                            gcs_url = retrieved_url # 只有非空字符串才赋值
                            print(f"  [成功] ID '{item_id}': 找到 gcs_url '{gcs_url}'")
                        else:
                            print(f"  [跳过] ID '{item_id}': 文档存在但 'gcs_url' 字段为空或不存在。")
                    else:
                        # 文档在Firestore中不存在
                        print(f"  [跳过] ID '{item_id}': 在集合中未找到文档。")
                        # gcs_url 保持为 None

                except Exception as e:
                    # 处理查询过程中可能出现的其他 Firestore 错误
                    print(f"  [错误] ID '{item_id}': 检索数据时出错: {e}。跳过此项。")
                    # gcs_url 保持为 None

                # --- 条件性添加 ---
                # 只有当 gcs_url 不是 None (即成功获取了非空字符串URL) 时才添加
                if gcs_url:
                    enriched_item = item.copy() # 创建原始字典的浅拷贝
                    enriched_item['gcs_url'] = gcs_url # 添加 gcs_url 字段
                    output_list.append(enriched_item) # 将处理后的字典添加到输出列表
                # else: # 如果 gcs_url 是 None，则忽略此项，不添加到输出列表

            else:
                 # item中有'id'但不是有效字符串
                 if 'id' in item: # 只有当id键存在但无效时才打印警告
                      print(f"  [跳过] 发现无效 ID: {item.get('id')} (类型: {type(item.get('id'))})。")
    logging.info(f"检索完成。共找到 {len(output_list)} 个包含有效 gcs_url 的项。")
    return output_list

# --- 路由 ---
@app.route('/')
def index():
    """主页路由"""
    return "Hello, Production World!"

# --- gemini分析图片路由 ---
@app.route('/gemini', methods=['POST'])
def gemini():

    if not request.is_json:
        return jsonify({"error": "请求 Content-Type 必须是 application/json"}), 415

    # 解析请求 JSON
    try:
        data = request.get_json()
        gcs_url = data.get('gcs_url')

        if not gcs_url or not gcs_url.startswith('gs://'):
            return jsonify({"error": "请求 JSON 中必须包含有效的 'gcs_url' (以 gs:// 开头)"}), 400

    except Exception as e:
        logging.error(f"解析请求 JSON 时出错: {e}")
        return jsonify({"error": f"无效的 JSON 请求: {e}"}), 400
    
    # gemini分析图片
    try:
        resp_gemini = analyze_image_with_gemini_fallback(gcs_url=gcs_url)
        return jsonify(resp_gemini), 200
    except Exception as e:
        logging.error(f"gemini分析图片时出错: {e}")
        return jsonify({"error": f"gemini分析图片时出错: {e}"}), 500

# --- 图片嵌入路由 ---
@app.route('/image-embedding', methods=['POST'])
def image_embedding():
    try:
        data = request.get_json()
        gcs_url = data.get('gcs_url')

        if not gcs_url or not gcs_url.startswith('gs://'):
            return jsonify({"error": "请求 JSON 中必须包含有效的 'gcs_url' (以 gs:// 开头)"}), 400

    except Exception as e:
        logging.error(f"解析请求 JSON 时出错: {e}")
        return jsonify({"error": f"无效的 JSON 请求: {e}"}), 400
    
    try:
        # 调用多模态嵌入函数
        resp_imageemb = image_multimodalembedding(image_path=gcs_url, model=mm_model)

        # 检查是否为 list
        if not isinstance(resp_imageemb, list):
            raise ValueError("多模态嵌入函数返回值格式错误，必须为列表")

        return jsonify({"image_emb": resp_imageemb}), 200

    except Exception as e:
        logging.error(f"生成多模态嵌入时出错: {e}, 请求的 gcs_url: {gcs_url}")
        return jsonify({"error": f"生成多模态嵌入时出错: {e}"}), 500
    
# --- 文本嵌入查询路由 ---
@app.route('/search', methods=['POST', 'OPTIONS'])
def text_to_search():
    if request.method == 'OPTIONS':
        # 处理预检请求，返回 200 并设置 CORS 头
        response = jsonify({'message': 'CORS preflight OK'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        return response, 200
    
    if request.method == 'POST':
        try:
            # 最好先检查 Content-Type
            if not request.is_json:
                 logging.warning("收到非 application/json Content-Type 的 POST 请求")
                 return jsonify({"error": "请求必须是 JSON (Content-Type: application/json)"}), 415

            data = request.get_json()
            if data is None: # get_json() 在 Content-Type 不对或body为空时可能返回 None
                 logging.warning("无法从请求中解析 JSON 数据或数据为空")
                 return jsonify({"error": "无效的 JSON 请求或请求体为空"}), 400

            search_text = data.get('text')
            if not search_text: # 检查 'text' 字段是否存在且不为空
                 logging.warning("请求 JSON 中缺少 'text' 字段或其值为空")
                 return jsonify({"error": "请求 JSON 中必须包含 'text' 字段且不能为空"}), 400

        except Exception as e:
            logging.error(f"解析请求 JSON 时出错: {e}", exc_info=True) # 记录详细错误
            return jsonify({"error": f"无效的 JSON 请求: {e}"}), 400
    
    try:
        # 调用多模态嵌入函数
        resp_textemb = text_multimodalembedding(text=search_text, model=mm_model)
    except Exception as e:
        logging.error(f"生成多模态嵌入时出错: {e}, 请求的 Text: {search_text}")
        return jsonify({"error": f"生成多模态嵌入时出错: {e}"}), 500
    
    try:
        # 查询
        resp_search = vector_search_find_neighbors(queries=resp_textemb, num_neighbors=NUM_NEIGHBORS)

        id_list = convert_match_neighbors_to_list_of_dicts(match_neighbor_data=resp_search)
        print(id_list)
        # return jsonify({"id_list": id_list}), 200

    except Exception as e:
        logging.error(f"Vector查询时出错: {e}")
        return jsonify({"error": f"Vector查询时出错: {e}"}), 500
    
    try:
        if firestore_db and COLLECTION_NAME:
            # 查询 Firestore 数据库
            logging.info(f"正在查询 Firestore 数据库...")
            valid_items = get_valid_gcs_url_items(id_list, firestore_db, COLLECTION_NAME)
            return jsonify(valid_items), 200

        else:
            return jsonify({"error": "Firestore 客户端初始化失败"}), 500
    except Exception as e:
        logging.error(f"Firestore查询时出错: {e}")
        return jsonify({"error": f"Firestore查询时出错: {e}"}), 500

# --- 图片嵌入查询路由 ---
@app.route('/imagesearch', methods=['POST', 'OPTIONS'])
def image_to_search():
    if request.method == 'OPTIONS':
        # 处理预检请求，返回 200 并设置 CORS 头
        response = jsonify({'message': 'CORS preflight OK'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        return response, 200
    
    if request.method == 'POST':
        try:
            logging.info(f"收到 POST 请求，正在解析数据...")
            # 检查 Content-Type 是否为 multipart/form-data
            content_type = request.content_type
            if not content_type or 'multipart/form-data' not in content_type:
                logging.warning(f"收到非 multipart/form-data 的 POST 请求，Content-Type: {content_type}")
                return jsonify({"error": "请求必须是 multipart/form-data 格式"}), 415

            # 检查是否上传了文件
            if 'image' not in request.files:
                logging.warning("请求中缺少 'image' 文件字段")
                return jsonify({"error": "请求必须包含名为 'image' 的文件字段"}), 400

            image_file = request.files['image']
            filename = image_file.filename

            # 检查文件名是否合法
            if not filename or not allowed_file(filename):
                logging.warning(f"上传的文件扩展名不支持: {filename}")
                return jsonify({"error": f"不支持的文件类型: {filename}"}), 415

            # 检查文件大小（双重校验）
            if image_file.content_length > MAX_IMAGE_SIZE_BYTES:
                logging.warning(f"上传的文件大小超过限制: {image_file.content_length / (1024 * 1024):.2f} MB")
                return jsonify({
                    "error": f"文件大小超过 {MAX_IMAGE_SIZE_BYTES / (1024 * 1024):.2f} MB 限制"}), 413
        except Exception as e:
            logging.error(f"解析请求时出错: {e}", exc_info=True) # 记录详细错误
            return jsonify({"error": f"无效的请求: {e}"}), 400
    
    try:
        # 生成临时文件
        logging.info(f"正在生成临时文件...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{filename.rsplit('.', 1)[1].lower()}") as temp_file:
            image_file.save(temp_file.name)
            temp_file_path = temp_file.name

        # 调用多模态嵌入函数获取嵌入
        logging.info(f"正在生成多模态嵌入...")
        resp_textemb = image_multimodalembedding(image_path=temp_file_path, model=mm_model)

        # 删除临时文件
        os.unlink(temp_file_path)
    except Exception as e:
        # 异常处理时确保临时文件被删除
        if 'temp_file_path' in locals():
            try:
                os.unlink(temp_file_path)
            except:
                pass
        logging.error(f"生成多模态嵌入时出错: {e}, 请求的 image: {temp_file_path}")
        return jsonify({"error": f"生成多模态嵌入时出错: {e}"}), 500
    
    try:
        resp_search = vector_search_find_neighbors(queries=resp_textemb, num_neighbors=NUM_NEIGHBORS)
        logging.info(f"已成功查询")
        id_list = convert_match_neighbors_to_list_of_dicts(match_neighbor_data=resp_search)
        # print(id_list)
        # return jsonify({"id_list": id_list}), 200

    except Exception as e:
        logging.error(f"Vector Search查询时出错: {e}")
        return jsonify({"error": f"Vector Search查询时出错: {e}"}), 500
    
    try:
        if firestore_db and COLLECTION_NAME:
            # 查询 Firestore 数据库
            valid_items = get_valid_gcs_url_items(id_list, firestore_db, COLLECTION_NAME)
            return jsonify(valid_items), 200
        else:
            return jsonify({"error": "Firestore 客户端初始化失败"}), 500
    except Exception as e:
        logging.error(f"Firestore查询时出错: {e}")
        return jsonify({"error": f"Firestore查询时出错: {e}"}), 500
    
# --- 数据入库路由 ---
@app.route('/upsert-datapoint', methods=['POST'])
def upsert_datapoint():

    if not request.is_json:
        return jsonify({"error": "请求 Content-Type 必须是 application/json"}), 415

    # 解析请求 JSON
    try:
        data = request.get_json()
        gcs_url = data.get('gcs_url')

        if not gcs_url or not gcs_url.startswith('gs://'):
            return jsonify({"error": "请求 JSON 中必须包含有效的 'gcs_url' (以 gs:// 开头)"}), 400

    except Exception as e:
        logging.error(f"解析请求 JSON 时出错: {e}")
        return jsonify({"error": f"无效的 JSON 请求: {e}"}), 400
    
    # 1) gemini分析图片
    try:
        resp_gemini = analyze_image_with_gemini_fallback(gcs_url=gcs_url)
        # 取出总的描述
        summery = resp_gemini['summary']

    except Exception as e:
        logging.error(f"gemini分析图片时出错: {e}")
        return jsonify({"error": f"gemini分析图片时出错: {e}"}), 500
    
    # try:
    #     vertexai.init(project=GCP_PROJECT, location=FALLBACK_REGION)
    #     mm_model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
    # except Exception as e:
    #     logging.error(f"初始化多模态嵌入模型时出错: {e}")
    #     return jsonify({"error": f"初始化多模态嵌入模型时出错: {e}"}), 500
    
    # 2) 图片多模态嵌入
    try:
        # 调用多模态嵌入函数
        resp_imageembdding = image_multimodalembedding(gcs_url=gcs_url, model=mm_model)
        # 检查是否为 list
        # if not isinstance(resp, list):
        #     raise ValueError("多模态嵌入函数返回值格式错误，必须为列表")

        # return jsonify({"image_emb": resp}), 200

    except Exception as e:
        logging.error(f"生成多模态嵌入时出错: {e}, 请求的 gcs_url: {gcs_url}")
        return jsonify({"error": f"生成多模态嵌入时出错: {e}"}), 500
    
    # 3）双写数据库
    unique_id = get_unique_id()
    json_data = {
        "id": unique_id,
        "gcs_url": gcs_url,
        "summery": summery
    }
    data_points = [
        IndexDatapoint(
            datapoint_id=unique_id,
            feature_vector=resp_imageembdding
        ),
    ]
    try:
        insert_info_embdding_data(json_data=json_data, unique_id=unique_id, data_point=data_points)
        return jsonify({"message": "数据已成功插入"}), 200
    
    except Exception as e:
        logging.error(f"插入数据时出错: {e}")
        return jsonify({"error": f"插入数据时出错: {e}"}), 500

# --- 主路由 ---
@app.route('/main', methods=['POST'])
def main():

    if not request.is_json:
        return jsonify({"error": "请求 Content-Type 必须是 application/json"}), 415

    # 解析请求 JSON
    try:
        data = request.get_json()
        gcs_url = data.get('gcs_url')

        if not gcs_url or not gcs_url.startswith('gs://'):
            return jsonify({"error": "请求 JSON 中必须包含有效的 'gcs_url' (以 gs:// 开头)"}), 400

    except Exception as e:
        logging.error(f"解析请求 JSON 时出错: {e}")
        return jsonify({"error": f"无效的 JSON 请求: {e}"}), 400
    
    # 1) gemini分析图片
    try:
        resp_gemini = analyze_image_with_gemini_fallback(gcs_url=gcs_url)
        # 取出总的描述
        summery = resp_gemini['summary']

    except Exception as e:
        logging.error(f"gemini分析图片时出错: {e}")
        return jsonify({"error": f"gemini分析图片时出错: {e}"}), 500
    
    # try:
    #     vertexai.init(project=GCP_PROJECT, location=FALLBACK_REGION)
    #     mm_model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
    # except Exception as e:
    #     logging.error(f"初始化多模态嵌入模型时出错: {e}")
    #     return jsonify({"error": f"初始化多模态嵌入模型时出错: {e}"}), 500
    
    # 2) 图片多模态嵌入
    try:
        # 调用多模态嵌入函数
        resp_imageembdding = image_multimodalembedding(gcs_url=gcs_url, model=mm_model)
        # 检查是否为 list
        # if not isinstance(resp, list):
        #     raise ValueError("多模态嵌入函数返回值格式错误，必须为列表")

        # return jsonify({"image_emb": resp}), 200

    except Exception as e:
        logging.error(f"生成多模态嵌入时出错: {e}, 请求的 gcs_url: {gcs_url}")
        return jsonify({"error": f"生成多模态嵌入时出错: {e}"}), 500
    
    # 3）双写数据库
    unique_id = get_unique_id()
    json_data = {
        "id": unique_id,
        "gcs_url": gcs_url,
        "summery": summery
    }
    data_points = [
        IndexDatapoint(
            datapoint_id=unique_id,
            feature_vector=resp_imageembdding
        ),
    ]
    try:
        insert_info_embdding_data(json_data=json_data, unique_id=unique_id, data_point=data_points)
        return jsonify({"message": "数据已成功插入"}), 200
    
    except Exception as e:
        logging.error(f"插入数据时出错: {e}")
        return jsonify({"error": f"插入数据时出错: {e}"}), 500

# --- 运行 ---
# 注意：我们不再使用 app.run()。Gunicorn 会负责导入并运行 'app' 这个 Flask 实例。
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=False, host='0.0.0.0', port=port)