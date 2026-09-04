# SPEC: K2PN – Giao diện tìm kiếm Video/Ảnh (React)

> Dự án: Cuộc thi Ho Chi Minh AIC
> Mục tiêu: Xây dựng UI React cho công cụ tìm kiếm video bằng nhiều phương thức (Text, ASR, OCR, Object), hiển thị kết quả dạng lưới ảnh và xem chi tiết từng kết quả.

---

## 1. Tổng quan bố cục

```
┌──────────────────────────────────────────────────────────────────────┐
│  Header: Logo | (Settings) (Refresh) | U                             │
├───────────────┬──────────────────────────────────────────────────────┤
│               │                                                      │
│  Sidebar trái │              Result Grid (ảnh/frame kết quả)         │
│  (Search      │                                                      │
│   Panels)     │                                                      │
│               │                                                      │
│  [+] [-]      │                                                      │
└───────────────┴──────────────────────────────────────────────────────┘
```

Click vào 1 ảnh trong Result Grid → mở modal hiển thị chi tiết ảnh đó.

---

## 2. Header

| Phần tử | Hành vi |
|---|---|
| Logo "K2PN" | Tĩnh, không cần logic |
| Icon Settings (bánh răng) | **Chưa cần chức năng** – render nhưng disable/placeholder, click không làm gì |
| Icon Refresh | Reset toàn bộ state ứng dụng về mặc định: xóa hết các Search Panel đang có, đưa về **1 panel mặc định rỗng** (type = text, query rỗng, toggle bật), đồng thời xóa kết quả đang hiển thị trong Result Grid |
| Chữ "U" | Hiển thị tĩnh, **không có hành vi** |

---

## 3. Sidebar trái – Danh sách Search Panel

Có thể có **nhiều Search Panel xếp chồng theo chiều dọc**. Mỗi panel là một khối truy vấn độc lập, đại diện cho 1 điều kiện tìm kiếm.

### 3.1 Cấu trúc 1 Search Panel

```
┌─────────────────────────────┐
│ [Tr] [🔊] [👤] [🖼]          │  ← 4 icon chọn loại tìm kiếm
│ ┌─────────────────────────┐ │
│ │ Textarea nhập truy vấn  │ │
│ │                         │ │
│ └─────────────────────────┘ │
│ (toggle)                    │  ← toggle bật/tắt panel
└─────────────────────────────┘
```

- **4 icon loại tìm kiếm** (góc trên panel): đại diện cho
  1. `Text` – tìm theo văn bản mô tả (icon "Tr")
  2. `ASR` – tìm theo lời thoại/âm thanh được nhận dạng (icon loa)
  3. `OCR` – tìm theo chữ xuất hiện trong khung hình (icon người/tài liệu – theo ảnh gốc, đặt tên `OCR`)
  4. `Object` – tìm theo vật thể xuất hiện trong khung hình (icon ảnh)

  → Chỉ **1 loại được active tại một thời điểm cho mỗi panel** (giống radio button). Click vào icon nào thì icon đó được highlight, các icon còn lại về trạng thái mặc định. State panel lưu `type: 'text' | 'asr' | 'ocr' | 'object'`.

- **Textarea**: nhập nội dung truy vấn tương ứng với loại đã chọn. Placeholder gợi ý theo `type` (VD: "Mô tả nội dung video bằng văn bản...").

- **Toggle (bên dưới panel)**: bật/tắt panel này có được tính vào lượt tìm kiếm tổng hay không. Panel tắt (off) vẫn giữ nguyên nội dung nhưng không gửi vào query. Đây là điều khiển chức năng duy nhất ở phần dưới panel (không còn icon nhỏ hay số đếm nào khác).

### 3.2 Nút thêm / xóa panel

- Nút **"+"** (màu xanh, cuối danh sách): thêm 1 Search Panel mới (mặc định: type = text, query rỗng, toggle bật, số đếm = 0) vào cuối danh sách.
- Nút **"-"** (màu đỏ, cạnh nút "+"): xóa panel cuối cùng trong danh sách. Nếu chỉ còn 1 panel thì disable nút "-" (không cho xóa hết, luôn giữ tối thiểu 1 panel).

---

## 4. Khu vực chính – Result Grid

- Hiển thị lưới các ảnh/thumbnail kết quả trả về từ truy vấn (frame trích từ video).
- Layout: grid responsive, khoảng **6–7 cột** trên desktop (theo ảnh mẫu), tự động xuống dòng.
- Mỗi ô là 1 ảnh thumbnail, có thể có overlay nhỏ (ví dụ logo kênh/label) nếu dữ liệu có – phần này lấy trực tiếp từ `thumbnailUrl` của item, không cần thiết kế riêng.
- **Click vào 1 ảnh** → mở **Modal chi tiết** hiển thị thông tin của kết quả đó:
  - Ảnh/frame phóng to
  - Thông tin: tên video, thời điểm (timestamp) trong video, mã frame, điểm số liên quan (score) nếu có, loại truy vấn đã match (text/asr/ocr/object)
  - Nút đóng modal

---

## 5. Cấu trúc dữ liệu (state)

```ts
type SearchType = 'text' | 'asr' | 'ocr' | 'object';

interface SearchPanel {
  id: string;
  type: SearchType;
  query: string;
  enabled: boolean;   // trạng thái toggle
}

interface ResultItem {
  id: string;
  thumbnailUrl: string;
  videoTitle: string;
  videoId: string;
  timestamp: string;   // ví dụ "00:12:35"
  frameId: string;
  score?: number;
  matchedType?: SearchType;
}

interface AppState {
  panels: SearchPanel[];       // luôn có >= 1 phần tử
  results: ResultItem[];
}
```

Giá trị mặc định khi reset (nhấn nút Refresh trên header):

```ts
{
  panels: [{ id: uuid(), type: 'text', query: '', enabled: true }],
  results: []
}
```

---

## 6. Cây component đề xuất

```
App
├── Header
│     ├── Logo
│     ├── SettingsButton      (disabled/placeholder)
│     ├── RefreshButton       (reset state)
│     └── StaticULabel
├── Sidebar
│     ├── SearchPanelList
│     │     └── SearchPanel (lặp theo panels[])
│     │           ├── SearchTypeIcons (4 icon: Text/ASR/OCR/Object)
│     │           ├── QueryTextarea
│     │           └── EnableToggle
│     └── PanelControls (nút + / -)
└── MainContent
      ├── ResultGrid
      │     └── ResultThumbnail (lặp theo results[])
      └── ResultDetailModal (hiện khi có ảnh được chọn)
```

---

## 7. Hành vi tổng hợp (tóm tắt nhanh)

| Hành động | Kết quả |
|---|---|
| Click icon loại tìm kiếm trong 1 panel | Đổi `type` của panel đó, highlight icon được chọn |
| Nhập chữ vào textarea | Cập nhật `query` của panel |
| Gạt toggle | Bật/tắt `enabled` của panel (panel tắt không tính vào tìm kiếm) |
| Click "+" | Thêm 1 panel mới mặc định vào cuối danh sách |
| Click "-" | Xóa panel cuối cùng (không xóa nếu chỉ còn 1 panel) |
| Click icon Refresh (header) | Reset toàn bộ `panels` về 1 panel mặc định, xóa `results` |
| Click icon Settings (header) | Không làm gì (placeholder) |
| Click chữ "U" | Không làm gì (hiển thị tĩnh) |
| Click 1 ảnh trong Result Grid | Mở modal chi tiết của ảnh đó |
| Đóng modal | Đóng modal, không thay đổi state khác |

---

## 8. Công nghệ & lưu ý triển khai

- **React** (function components + hooks: `useState`, có thể dùng `useReducer` cho state của `panels`/`results` nếu logic phức tạp dần).
- Styling: Tailwind CSS (hoặc CSS module tương đương), theo phong cách gọn – nền trắng, sidebar viền mỏng, icon dạng outline như ảnh gốc.
- Dữ liệu kết quả (`results`) ban đầu dùng **mock data tĩnh** (mảng JSON ảnh mẫu) để dựng UI; phần gọi API tìm kiếm thật sẽ nối vào sau, không nằm trong phạm vi bản đặc tả này.
- Modal chi tiết ảnh chỉ cần render nội dung tĩnh từ `ResultItem` được click, không cần gọi thêm API.

---

## 9. Ngoài phạm vi (Out of scope – chưa cần làm)

- Chức năng thật của nút Settings.
- Logic thật đứng sau chữ "U" ở header.
- Gọi API tìm kiếm thật / kết nối backend AIC.
- Responsive cho mobile (ưu tiên desktop trước, theo ảnh mẫu gốc).