# Visual Question Answering (VQA) – COCO subset

Dự án này triển khai bài toán **Visual Question Answering (VQA)**: cho một **ảnh** và một **câu hỏi**, mô hình dự đoán **câu trả lời** (phân loại theo tập answer xuất hiện trong tập train).

Repo hiện tập trung vào 2 hướng tiếp cận (đều nằm trong notebook):

- `VQA_CNN+LSTM.ipynb`: **ResNet50 (timm) + Embedding + BiLSTM** để mã hoá câu hỏi, sau đó **phân loại** câu trả lời.
- `VQA_Transformers.ipynb`: **ViT (vision transformer) + RoBERTa** để mã hoá ảnh/câu hỏi, ghép đặc trưng và **phân loại**.

## Cấu trúc thư mục

- `VQA_CNN+LSTM.ipynb`: Notebook huấn luyện/evaluate mô hình CNN+LSTM.
- `VQA_Transformers.ipynb`: Notebook huấn luyện/evaluate mô hình Transformer-based.
- `vqa_coco_dataset/`
  - `val2014-resised/`: ảnh (đuôi `.jpg`) đã resize sẵn (8127 ảnh).
  - `vaq2.0.TrainImages.txt`: 7846 mẫu train.
  - `vaq2.0.DevImages.txt`: 1952 mẫu validation/dev.
  - `vaq2.0.TestImages.txt`: 2022 mẫu test.

## Dataset format

Các file `vaq2.0.*Images.txt` có dạng mỗi dòng:

`<image_path>\t<question>?<answer>`

Notebook đang parse bằng cách tách theo `\t`, sau đó tách phần QA theo dấu `?` để lấy:

- `image_path`: tên file ảnh (trỏ vào thư mục ảnh).
- `question`: câu hỏi (được thêm lại dấu `?`).
- `answer`: câu trả lời dạng text.

## Pipeline (tóm tắt theo notebook)

### 1) CNN + LSTM (`VQA_CNN+LSTM.ipynb`)

- **Text**
  - Tokenize bằng `spacy` (`en_core_web_sm`)
  - Build vocab bằng `torchtext` (`min_freq=2`, có các token đặc biệt `<pad> <sos> <eos> <unk>`)
  - Pad/truncate câu hỏi về `max_seq_len=20`
- **Image**
  - Transform: resize `224x224`, normalize theo ImageNet mean/std
  - Encoder ảnh: `timm.create_model('resnet50', pretrained=True, num_classes=hidden_size)`
- **Model**
  - Embedding → BiLSTM → lấy timestep cuối
  - Ghép `img_features` + `lstm_out`
  - Qua BiLSTM thứ 2 + dropout + linear ra `n_classes`
- **Train/Eval**
  - Loss: `CrossEntropyLoss`
  - Optimizer: Adam, `lr=1e-2`, `epochs=50`, `StepLR(gamma=0.1)`

### 2) Transformers (`VQA_Transformers.ipynb`)

- **Text encoder**: `RobertaModel.from_pretrained('roberta-base')` → `pooler_output`
- **Visual encoder**: `ViTModel.from_pretrained('google/vit-base-patch16-224')` → `pooler_output`
- **Feature extractor / tokenizer**
  - Ảnh: `ViTImageProcessor.from_pretrained(...)`
  - Câu hỏi: `AutoTokenizer.from_pretrained('roberta-base')` (pad/truncate `max_length=20`)
- **Classifier**
  - Ghép vector text+image (kích thước 768*2)
  - Qua BiLSTM + dropout + linear ra `n_classes`
- **Freeze**
  - Notebook có hàm `model.freeze()` để freeze encoder (mặc định freeze cả visual + textual, train classifier)

## Kết quả tham khảo (từ output trong notebook)

Lưu ý: đây là **accuracy theo cách label hoá answer** trong notebook (phân loại theo tập answer trong train), nên chỉ dùng để tham khảo.

- `VQA_CNN+LSTM.ipynb`
  - Val accuracy: `~0.5359`
  - Test accuracy: `~0.5490`
- `VQA_Transformers.ipynb`
  - Val accuracy: `~0.6547`
  - Test accuracy: `~0.6365`

## Cách chạy

### Option A — Chạy trên Google Colab (đúng như notebook gốc)

1. Mở notebook trên Colab.
2. Tải dataset theo cell “Download dataset” (dùng `gdown`) hoặc upload thủ công lên Google Drive rồi copy vào runtime.
3. Chạy lần lượt các cell theo thứ tự.

### Option B — Chạy local bằng Jupyter

Yêu cầu:

- Python 3.9+ (khuyến nghị 3.10/3.11)
- Jupyter Notebook/Lab

Gợi ý cài dependency (tuỳ môi trường CUDA/CPU bạn đang dùng):

```bash
pip install jupyter torch torchvision torchtext timm transformers spacy scikit-learn pillow matplotlib pandas numpy
python -m spacy download en_core_web_sm
```

Sau đó mở Jupyter và chạy:

```bash
jupyter notebook
```

### Lưu ý về đường dẫn (quan trọng)

Notebook được viết theo ngữ cảnh Colab nên có vài chỗ hard-code path:

- `VQA_CNN+LSTM.ipynb` mặc định `root_dir='/content/val2014-resised/'`
  - Khi chạy local, hãy đổi thành `root_dir='vqa_coco_dataset/val2014-resised/'`
  - Và đảm bảo các file `vaq2.0.*Images.txt` được trỏ đúng (hiện nằm trong `vqa_coco_dataset/`)
- `VQA_Transformers.ipynb` đang đọc txt từ `datasets/vaq2.0.*Images.txt`
  - Khi chạy local trong repo này, hãy đổi thành `vqa_coco_dataset/vaq2.0.*Images.txt`

## Known issues / cần chỉnh nhẹ khi chạy local

- `VQA_Transformers.ipynb` có cell khởi tạo dataset truyền tham số `label_encoder=label_encoder` nhưng trong class `VQADataset` không nhận tham số này và biến `label_encoder` cũng không được định nghĩa. Khi chạy, bạn có thể **xoá các dòng `label_encoder=label_encoder`** ở chỗ tạo `VQADataset(...)`.

