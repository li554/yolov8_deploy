import os
from tqdm import tqdm
from prune import prune_model
from relation import find_parent_nodes, visualize_nodes, metric
from ultralytics import YOLO


class PruneModel:
    def __init__(self, weights="weights/last.pt"):
        # Load a model
        self.yolo = YOLO(weights)

    def prune(self, factor=0.7, save_dir="weights/prune.pt"):
        prune_model(self.yolo, save_dir, factor)

    def train(self, save_dir="weights/retrain.pt"):
        self.yolo.train(data='diagram.yaml', Distillation=None, loss_type='None', amp=False, imgsz=640,
                        epochs=50, batch=20, device=1, workers=4, name="default")
        self.yolo.save(save_dir)

    def sparse_train(self, save_dir='weight/sparse.pt'):
        self.yolo.train(data='diagram.yaml', Distillation=None, loss_type='sparse', amp=False, imgsz=640,
                        epochs=50, batch=20, device=0, workers=4, name="sparse")
        self.yolo.save(save_dir)

    def distill(self, t_weight, loss_type='mgd', save_dir="weights/distill.pt"):
        model_t = YOLO(t_weight)
        self.yolo.train(data="diagram.yaml", Distillation=model_t.model, loss_type=loss_type, amp=False, imgsz=640,
                        epochs=100, batch=20, device=0, workers=4, lr0=0.001)
        self.yolo.save(save_dir)

    def export(self, **kwargs):
        self.yolo.export(**kwargs)

    @staticmethod
    def compare(weights=None):
        # 统计压缩前后的参数量，精度，计算量
        if weights is None:
            weights = []
        results = []
        for weight in weights:
            yolo = YOLO(weight)
            metric = yolo.val(data='diagram.yaml', imgsz=640)
            n_l, n_p, n_g, flops = yolo.info()
            acc = metric.box.map
            results.append((weight, n_l, n_p, n_g, flops, acc))
        for weight, layer, n_p, n_g, flops, acc in results:
            print(f"Weight: {weight}, Acc: {acc}, Params: {n_p}, FLOPs: {flops}")

    def predict(self, source):
        results = self.yolo.predict(source)[0]
        nodes = results.boxes.xyxy
        nodes = nodes.tolist()
        ori_img = results.orig_img
        parent_nodes = find_parent_nodes(nodes)
        visualize_nodes(ori_img, nodes, parent_nodes)

    def evaluate(self, data_path):
        bboxes_list = []
        pred_bboxes_list = []
        parent_ids_list = []
        pred_parent_ids_list = []

        imgs_path = os.path.join(data_path, "images/val")
        labels_path = os.path.join(data_path, "plabels/val")

        # 读取标注文件
        for img in tqdm(os.listdir(imgs_path)):
            img_path = os.path.join(imgs_path, img)

            # 检查文件后缀并构建相应的标注文件路径
            if img.endswith(".png"):
                label_path = os.path.join(labels_path, img.replace(".png", ".txt"))
            elif img.endswith(".webp"):
                label_path = os.path.join(labels_path, img.replace(".webp", ".txt"))
            else:
                continue

            with open(label_path, "r") as f:
                lines = f.readlines()

            results = self.yolo.predict(img_path)[0]
            pred_bboxes = results.boxes.xyxy
            pred_bboxes = pred_bboxes.tolist()
            pred_bboxes_list.append(pred_bboxes)
            pred_parent_ids = find_parent_nodes(pred_bboxes)
            pred_parent_ids_list.append(pred_parent_ids)
            ih, iw = results.orig_img.shape[:2]
            bboxes = []
            parent_ids = []
            for line in lines:
                line = line.strip().split()
                x, y, w, h, px, py, pw, ph, p = map(float, line[1:])
                x1, y1, x2, y2 = int((x - w / 2) * iw), int((y - h / 2) * ih), int((x + w / 2) * iw), int(
                    (y + h / 2) * ih)
                bboxes.append((x1, y1, x2, y2))
                parent_ids.append(int(p))
            bboxes_list.append(bboxes)
            parent_ids_list.append(parent_ids)
        precision, recall, f1_score = metric(bboxes_list, pred_bboxes_list, parent_ids_list, pred_parent_ids_list)
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1_score}")


if __name__ == '__main__':
    model = PruneModel("weights/yolov8n.pt")
    model.export(format="ncnn")
    # model.sparse_train("weights/sparse.pt")
    # model.prune(factor=0.2, save_dir="weights/prune.pt")
    # model.train()
    # model.distill("weights/sparse.pt", loss_type="mgd")
    # model.evaluate("datasets/diagram")
    # model.predict("datasets/diagram/images/val/0593.png")