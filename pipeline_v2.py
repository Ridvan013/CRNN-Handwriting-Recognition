import sys
import os
import time
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import cv2
import numpy as np
import math
import csv

# ==========================================
# Setup Paths
# ==========================================
# Add CRAFT to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
craft_path = os.path.join(script_dir, 'CRAFT-pytorch-master', 'CRAFT-pytorch-master', 'CRAFT-pytorch-master')
sys.path.append(craft_path)

try:
    from craft import CRAFT
    import craft_utils
    import imgproc
    import file_utils
except ImportError as e:
    print(f"Error importing CRAFT modules: {e}")
    print(f"Checked path: {craft_path}")
    sys.exit(1)

# ==========================================
# CRNN Model Definition (Standalone)
# ==========================================
CHAR_LIST = "!\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

def setup_gpu():
    """GPU setup"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✅ GPU Found: {torch.cuda.get_device_name(0)}")
        cudnn.benchmark = True
        return device
    else:
        print("⚠️ GPU not found, using CPU")
        return torch.device('cpu')

class CRNNModel(nn.Module):
    def __init__(self, img_height: int = 32, img_width: int = 128, num_classes: int = None):
        super(CRNNModel, self).__init__()
        
        if num_classes is None:
            num_classes = len(CHAR_LIST) + 1  # +1 for blank token
        
        self.num_classes = num_classes
        
        # CNN Feature Extraction
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (16,64)
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (8,32)
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,1)),  # (4,32)
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,1)),  # (2,32)
            
            # Final conv
            nn.Conv2d(512, 512, kernel_size=2, padding=0),  # (1,31,512)
            nn.ReLU(inplace=True)
        )
        
        # RNN (Bidirectional LSTM)
        self.rnn = nn.Sequential(
            nn.LSTM(512, 256, bidirectional=True, batch_first=False, dropout=0.2),
            nn.LSTM(512, 256, bidirectional=True, batch_first=False, dropout=0.2)
        )
        
        # Classifier
        self.classifier = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # CNN forward
        conv_features = self.cnn(x)  # [B, 512, 1, W']
        
        # Reshape for RNN: [B, C, H, W] -> [W, B, C] (time-major)
        B, C, H, W = conv_features.size()
        
        # Squeeze height and permute for RNN
        rnn_input = conv_features.squeeze(2).permute(2, 0, 1)  # [W', B, C]
        
        # RNN forward
        lstm_out1, _ = self.rnn[0](rnn_input)
        lstm_out2, _ = self.rnn[1](lstm_out1)
        
        # Classifier
        logits = self.classifier(lstm_out2)  # [W', B, num_classes]
        
        # Log softmax for CTC
        log_probs = nn.functional.log_softmax(logits, dim=2)
        
        return log_probs

def beam_search_decode(log_probs, input_lengths, beam_width=10, blank_index=None, device='cuda'):
    """
    GPU-optimized beam search decoding for CTC predictions
    """
    if blank_index is None:
        blank_index = len(CHAR_LIST)  # Blank token index
    
    seq_len, batch_size, num_classes = log_probs.shape
    results = []
    
    # Ensure log_probs is on the correct device
    if log_probs.device != device:
        log_probs = log_probs.to(device)
    
    # Convert input_lengths to tensor if needed
    if isinstance(input_lengths, torch.Tensor):
        input_lengths = input_lengths.to(device)
    else:
        input_lengths = torch.tensor(input_lengths, dtype=torch.long, device=device)
    
    for batch_idx in range(batch_size):
        seq_len_actual = int(input_lengths[batch_idx].item())
        pred = log_probs[:seq_len_actual, batch_idx, :]  # (seq_len_actual, num_classes)
        
        # Initialize beam
        beam = [{'sequence': [], 'log_prob': torch.tensor(0.0, device=device, dtype=log_probs.dtype), 'last_char': None}]
        
        for t in range(seq_len_actual):
            new_beam = []
            
            # Get top-k characters for this timestep
            top_k = min(beam_width * 2, num_classes)
            top_k_probs, top_k_indices = torch.topk(pred[t], top_k)
            
            for beam_item in beam:
                for k_idx in range(len(top_k_indices)):
                    char_idx = int(top_k_indices[k_idx].item())
                    char_log_prob = top_k_probs[k_idx]
                    
                    new_log_prob = beam_item['log_prob'] + char_log_prob
                    
                    if char_idx == blank_index:
                        new_beam.append({
                            'sequence': beam_item['sequence'].copy(),
                            'log_prob': new_log_prob,
                            'last_char': None
                        })
                    elif beam_item['last_char'] == char_idx:
                        new_beam.append({
                            'sequence': beam_item['sequence'].copy(),
                            'log_prob': new_log_prob,
                            'last_char': char_idx
                        })
                    else:
                        new_sequence = beam_item['sequence'].copy()
                        new_sequence.append(char_idx)
                        new_beam.append({
                            'sequence': new_sequence,
                            'log_prob': new_log_prob,
                            'last_char': char_idx
                        })
            
            # Keep only top beam_width items
            if len(new_beam) > beam_width:
                log_probs_tensor = torch.stack([item['log_prob'] for item in new_beam])
                _, top_indices = torch.topk(log_probs_tensor, beam_width)
                beam = [new_beam[int(idx.item())] for idx in top_indices]
            else:
                beam = new_beam
        
        # Get best sequence
        if beam:
            log_probs_final = torch.stack([item['log_prob'] for item in beam])
            best_idx = torch.argmax(log_probs_final).item()
            best_sequence = beam[best_idx]['sequence']
        else:
            best_sequence = []
        results.append(best_sequence)
    
    return results

# ==========================================
# Configuration
# ==========================================
DEVICE = setup_gpu()

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = {}
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

# ==========================================
# CRAFT Detection
# ==========================================
def load_craft_model(args):
    print(f"Loading CRAFT model from {args.craft_model}...")
    net = CRAFT()
    
    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.craft_model)))
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False
    else:
        net.load_state_dict(copyStateDict(torch.load(args.craft_model, map_location='cpu')))
    
    net.eval()
    return net

def run_craft_detection(net, image, args):
    """
    Run CRAFT detection on an image.
    Returns: boxes (list of polys), polys (list of polys)
    """
    t0 = time.time()
    
    # Resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(
        image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio
    )
    ratio_h = ratio_w = 1 / target_ratio

    # Preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    
    if args.cuda:
        x = x.cuda()

    # Forward pass
    with torch.no_grad():
        y, feature = net(x)

    # Make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(
        score_text, score_link, args.text_threshold, args.link_threshold, args.low_text, args.poly
    )

    # Coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    print(f"CRAFT Detection time: {time.time() - t0:.3f}s")
    return boxes, polys

# ==========================================
# CRNN Recognition
# ==========================================
def load_crnn_model(model_path, device):
    print(f"Loading CRNN model from {model_path}...")
    model = CRNNModel(img_height=32, img_width=128, num_classes=len(CHAR_LIST)+1)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f"Error: CRNN model not found at {model_path}")
        sys.exit(1)
        
    model.to(device)
    model.eval()
    return model

def preprocess_for_crnn(img_crop, device):
    """
    Preprocess a cropped word image for CRNN inference.
    Matches the training preprocessing in greedy.py IAMDataset.
    """
    # Convert to grayscale if needed
    if len(img_crop.shape) == 3:
        img_gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img_crop

    # Resize to height 32, keep aspect ratio, then pad/resize to width 128
    # Actually, greedy.py simply resizes to (32, 128) directly in the transform
    # Let's follow greedy.py's transform logic exactly:
    # 1. Invert colors (1.0 - x)
    # 2. Normalize ((x - 0.5) / 0.5)
    # 3. Resize to (32, 128)
    
    # Prepare tensor
    img_tensor = torch.from_numpy(img_gray).float() / 255.0 # [0, 1]
    
    # Invert colors (assuming white background, black text like IAM)
    # Check if image is already inverted or not. 
    # IAM dataset usually has white background. 
    # greedy.py does `1.0 - x`, which implies it expects white background (1.0) and wants black background (0.0) for sparse tensor?
    # Or maybe it just inverts it. Let's stick to the code.
    img_tensor = 1.0 - img_tensor
    
    # Normalize
    img_tensor = (img_tensor - 0.5) / 0.5
    
    # Add dimensions: [H, W] -> [1, 1, H, W] for interpolation
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
    
    # Resize to (32, 128)
    img_tensor = torch.nn.functional.interpolate(
        img_tensor, 
        size=(32, 128), 
        mode='bilinear', 
        align_corners=False
    )
    
    # Remove batch dim: [1, 1, 32, 128] -> [1, 32, 128]
    img_tensor = img_tensor.squeeze(0)
    
    return img_tensor.to(device)

def sort_boxes(boxes):
    """
    Sort boxes from top to bottom, left to right.
    """
    # Calculate centroids
    centroids = []
    for box in boxes:
        poly = np.array(box).astype(np.int32)
        x, y, w, h = cv2.boundingRect(poly)
        cx = x + w // 2
        cy = y + h // 2
        centroids.append([cx, cy, h, box])

    # Sort by Y first
    centroids.sort(key=lambda k: k[1])
    
    # Group by lines
    lines = []
    current_line = []
    if not centroids:
        return []
        
    current_line.append(centroids[0])
    
    # Threshold for line grouping (average height of current line)
    # We update this dynamically or use the first box's height
    
    for i in range(1, len(centroids)):
        cx, cy, h, box = centroids[i]
        
        # Get average y of current line
        avg_y = sum([c[1] for c in current_line]) / len(current_line)
        avg_h = sum([c[2] for c in current_line]) / len(current_line)
        
        # If current box's y is within a threshold of the line's average y
        # It belongs to the same line. Threshold: half of average height.
        if abs(cy - avg_y) < (avg_h * 0.5):
            current_line.append(centroids[i])
        else:
            # New line, sort the previous line by X
            current_line.sort(key=lambda k: k[0])
            lines.extend([c[3] for c in current_line])
            current_line = [centroids[i]]
            
    # Append last line
    if current_line:
        current_line.sort(key=lambda k: k[0])
        lines.extend([c[3] for c in current_line])
        
    return lines

def merge_boxes(boxes, x_threshold=30, y_threshold_ratio=0.5):
    """
    Merge boxes that are close to each other horizontally and aligned vertically.
    Fixes issues where single words are split into multiple boxes.
    """
    if not boxes:
        return []
        
    merged = []
    current_box = boxes[0]
    
    for i in range(1, len(boxes)):
        next_box = boxes[i]
        
        # Get bounding rects
        poly1 = np.array(current_box).astype(np.int32)
        x1, y1, w1, h1 = cv2.boundingRect(poly1)
        
        poly2 = np.array(next_box).astype(np.int32)
        x2, y2, w2, h2 = cv2.boundingRect(poly2)
        
        right1 = x1 + w1
        left2 = x2
        
        # Check vertical alignment (centers close)
        cy1 = y1 + h1/2
        cy2 = y2 + h2/2
        avg_h = (h1 + h2) / 2
        
        vertical_diff = abs(cy1 - cy2)
        horizontal_gap = left2 - right1
        
        # Conditions for merge:
        # 1. Vertically aligned (diff < half height)
        # 2. Horizontally close (gap < threshold)
        # 3. Next box is to the right of current box (not a new line)
        
        if vertical_diff < (avg_h * y_threshold_ratio) and horizontal_gap < x_threshold and horizontal_gap > -10:
            # Merge
            new_x = min(x1, x2)
            new_y = min(y1, y2)
            new_right = max(x1+w1, x2+w2)
            new_bottom = max(y1+h1, y2+h2)
            new_w = new_right - new_x
            new_h = new_bottom - new_y
            
            # Create new 4-point box
            current_box = np.array([
                [new_x, new_y],
                [new_x + new_w, new_y],
                [new_x + new_w, new_y + new_h],
                [new_x, new_y + new_h]
            ], dtype=np.float32)
        else:
            merged.append(current_box)
            current_box = next_box
            
    merged.append(current_box)
    return merged

def recognize_words(image, boxes, crnn_model, device):
    """
    Recognize text in each bounding box.
    """
    results = []
    
    # 1. Sort boxes first (Top-Down, Left-Right)
    boxes = sort_boxes(boxes)
    
    # 2. Merge close boxes (Fix split words)
    boxes = merge_boxes(boxes)
    
    print(f"Recognizing {len(boxes)} words...")
    
    for i, box in enumerate(boxes):
        # Box is 4x2 array [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        # Get bounding rect
        poly = np.array(box).astype(np.int32)
        x, y, w, h = cv2.boundingRect(poly)
        
        # Add some padding
        pad = 2
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(image.shape[1] - x, w + 2*pad)
        h = min(image.shape[0] - y, h + 2*pad)
        
        # Crop
        crop = image[y:y+h, x:x+w]
        
        if crop.size == 0:
            continue
            
        # Preprocess
        input_tensor = preprocess_for_crnn(crop, device) # [1, 32, 128]
        input_tensor = input_tensor.unsqueeze(0) # [1, 1, 32, 128] Batch size 1
        
        # Inference
        with torch.no_grad():
            log_probs = crnn_model(input_tensor) # [T, B, C]
            
        # Decode
        # Beam search expects input_lengths
        input_lengths = torch.IntTensor([log_probs.size(0)]).to(device)
        
        # Use beam search from greedy.py
        decoded_sequences = beam_search_decode(
            log_probs, 
            input_lengths, 
            beam_width=10, 
            blank_index=len(CHAR_LIST), 
            device=device
        )
        
        # Convert indices to string
        pred_text = ""
        if decoded_sequences:
            pred_indices = decoded_sequences[0]
            pred_text = "".join([CHAR_LIST[idx] for idx in pred_indices])
            
        results.append({
            'box': box,
            'text': pred_text,
            'confidence': 1.0 # Placeholder, beam search doesn't return conf easily here
        })
        
    return results

# ==========================================
# Visualization
# ==========================================
def visualize_results(image, results, output_path):
    vis_img = image.copy()
    
    for res in results:
        box = res['box']
        text = res['text']
        
        poly = np.array(box).astype(np.int32)
        
        # Draw box (Green)
        cv2.polylines(vis_img, [poly], True, (0, 255, 0), 2)
        
        # Draw text (Blue)
        # Calculate position
        x, y, w, h = cv2.boundingRect(poly)
        
        # Put text background for readability
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        
        # Draw text BELOW the box
        text_x = x
        text_y = y + h + 25  # Shift down by height + padding
        
        cv2.rectangle(vis_img, (text_x, text_y - text_h - 5), (text_x + text_w, text_y + 5), (255, 255, 255), -1)
        
        # Put text
        cv2.putText(vis_img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
    cv2.imwrite(output_path, vis_img)
    print(f"Saved result to {output_path}")

# ==========================================
# Main
# ==========================================
def main():
    # USER: Change this path to test different images
    # default_image_path = r"c:\Users\RIDVAN\Desktop\CRNN\CRNN_1\iam sentence\images\img_00001.png"
    # Using relative path for portability, but user can use absolute
    default_image_path = os.path.join("iam sentence", "images", "img_00034.png")
    
    parser = argparse.ArgumentParser(description='CRAFT + CRNN Pipeline')
    parser.add_argument('--image', type=str, default=default_image_path, help='Path to input image')
    parser.add_argument('--craft_model', default='CRAFT-pytorch-master/CRAFT-pytorch-master/CRAFT-pytorch-master/craft_mlt_25k.pth', type=str, help='Path to CRAFT model')
    parser.add_argument('--crnn_model', default='Model/best_model_wa.pth', type=str, help='Path to CRNN model')
    parser.add_argument('--text_threshold', default=0.6, type=float, help='text confidence threshold')
    parser.add_argument('--low_text', default=0.35, type=float, help='text low-bound score')
    parser.add_argument('--link_threshold', default=0.2, type=float, help='link confidence threshold')
    parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda')
    parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
    parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
    parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
    
    args = parser.parse_args()
    
    print(f"Processing image: {args.image}")
    
    # Check inputs
    if not os.path.exists(args.image):
        print(f"Error: Image not found at {args.image}")
        # Try absolute path if relative fails
        abs_path = os.path.abspath(args.image)
        if os.path.exists(abs_path):
            args.image = abs_path
            print(f"Found at absolute path: {args.image}")
        else:
            print(f"Also checked absolute path: {abs_path}")
            sys.exit(1)
        
    # Load Models
    craft_net = load_craft_model(args)
    crnn_model = load_crnn_model(args.crnn_model, DEVICE)
    
    # Load Image
    image = imgproc.loadImage(args.image)
    
    # 1. Detect
    print("Running Text Detection...")
    boxes, polys = run_craft_detection(craft_net, image, args)
    print(f"Detected {len(boxes)} text regions.")
    
    # 2. Recognize
    print("Running Text Recognition...")
    # Reload image with cv2 to get raw array for cropping (imgproc.loadImage normalizes it?)
    # imgproc.loadImage returns numpy array. Let's use it.
    # Note: imgproc.loadImage reads via skimage or cv2.imread. 
    # It returns RGB if skimage, BGR if cv2. 
    # imgproc.loadImage implementation:
    #   img = skimage.io.imread(fname)
    #   if img.shape[0] == 2: img = img[0]
    #   if len(img.shape) == 2 : img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    #   if img.shape[2] == 4 : img = img[:,:,:3]
    #   img = np.array(img)
    # It returns RGB.
    
    # For visualization and cropping, let's ensure we have a standard BGR image for OpenCV
    image_cv = cv2.imread(args.image)
    
    results = recognize_words(image_cv, boxes, crnn_model, DEVICE)
    
    # 3. Visualize
    filename = os.path.basename(args.image)
    name, ext = os.path.splitext(filename)
    output_path = f"result_pipeline_{name}.jpg"
    txt_output_path = f"result_pipeline_{name}.txt"
    
    visualize_results(image_cv, results, output_path)
    
    # 4. Save Text Results
    # Construct predicted sentence
    predicted_sentence = " ".join([res['text'] for res in results])
    
    # Load Ground Truth
    ground_truth = "Not found in CSV"
    csv_path = os.path.join("iam sentence", "IAM_Sentences_fixed.csv")
    if os.path.exists(csv_path):
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader) # Skip header
                for row in reader:
                    if len(row) >= 2:
                        if row[0] == filename:
                            ground_truth = row[1]
                            break
        except Exception as e:
            print(f"Error reading CSV: {e}")
            
    # Write to TXT
    with open(txt_output_path, 'w', encoding='utf-8') as f:
        f.write(f"Image: {filename}\n")
        f.write(f"Ground Truth: {ground_truth}\n")
        f.write(f"Prediction:   {predicted_sentence}\n")
        
    print(f"\n--- Results ---")
    print(f"Ground Truth: {ground_truth}")
    print(f"Prediction:   {predicted_sentence}")
    print(f"Saved text results to {txt_output_path}")
    
    print("Done.")

if __name__ == '__main__':
    main()
