import argparse
import time
from pathlib import Path
import math
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, non_max_suppression, apply_classifier, scale_coords, \
    xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


def detect(save_img=False):
    source, weights, view_img, save_img, save_txt, imgsz, save_xxyy, tile_x, tile_y, object_size_px = opt.source, opt.weights, opt.view_img, opt.save_img, opt.save_txt, opt.img_size,opt.save_xxyy,opt.tile_x, opt.tile_y, opt.object_size_px
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        #save_img = False
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
#    if device.type != 'cpu':
#        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    #length = math.ceil(opt.img_size/8)

    for path, img, im0s, vid_cap in dataset:
        
        img_height, img_width = img.shape[1:]
        # devide in to chunks
        n_x = math.ceil(img_width/tile_x)
        n_y = math.ceil(img_height/tile_y)
        height1=img_height/n_y
        width1=img_width/n_x
            
        while ((img_height-height1)<object_size_px/2):
              n_y=n_y+1
              height1=img_height/n_y
        
        while ((img_width-width1)<object_size_px/2):
              n_x=n_x+1
              width1=img_width/n_x
              
        offset_x=math.floor(img_width/n_x)
        offset_y=math.floor(img_height/n_y)
        
        # initialize out struct
        preds = []
        # loop to subset into each crop
        for i in range(n_y): 
            for j in range(n_x):
              
                #logic to make overlapping tiles after the first iteration
                if i==0:
                    offset_y_new=offset_y*i
                else:
                    offset_y_new=(offset_y*i)-(tile_y-offset_y)
                
                if j==0:
                    offset_x_new=offset_x*j
                else:
                    offset_x_new=(offset_x*j)-(tile_x-offset_x)
                                            
                # make crop
                cropped_img = torch.from_numpy(img[:, offset_y_new:min(offset_y_new+tile_y, img_height), offset_x_new:min(offset_x_new+tile_x, img_width)]).to(device)

        #       img = torch.from_numpy(img).to(device)
                cropped_img = cropped_img.half() if half else cropped_img.float()  # uint8 to fp16/32
                cropped_img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if cropped_img.ndimension() == 3:
                    cropped_img = cropped_img.unsqueeze(0)
        
                # Inference
                #t1 = time_synchronized()
                pred = model(cropped_img, augment=opt.augment)[0]

                if len(pred) > 0:
                    pred[:,:, 0] += offset_x_new
                    pred[:,:, 1] += offset_y_new
                    preds.append(pred)                
      
        # Apply NMS
        pred = non_max_suppression(torch.cat(preds, 1), opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        #t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            #txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            txt_path = str(save_dir / 'labels' / p.parent.name) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            if not Path(txt_path + '.csv').exists():
                with open(txt_path + '.csv', 'a') as f:
                    if save_xxyy:
                        f.write(("%s,%s,%s,%s,%s,%s,%s,%s,%s" % ('filename','class','x','y','w','h','height','width','box_confidence')+ '\n'))
                    else:
                        f.write(("%s,%s,%s,%s,%s,%s,%s,%s,%s" % ('filename','class','xmin','xmax','ymin','ymax','height','width','box_confidence')+ '\n'))

            #s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[1:], det[:, :4], im0.shape[:2]).round()

                # Print results
                #for c in det[:, -1].unique():
                #    n = (det[:, -1] == c).sum()  # detections per class
                #    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string


                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        if save_xxyy:
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (p.name, names[int(cls)], *xywh, gn[1],gn[0],conf) if opt.save_conf else (p.name,names[int(cls)], *xywh, gn[1], gn[0])  # label format
                            with open(txt_path + '.csv', 'a') as f: 
                                f.write(('%s, '+'%s, '+('%g, ' * (len(line)-3)+'%g').rstrip()) % line + '\n')
                        else:
                            xxyy=[(torch.tensor(xyxy).view(1, 4)/ gn).view(-1).tolist()[i] for i in [0,2,1,3]]
                            line = (p.name, names[int(cls)], *xxyy,gn[1],gn[0], conf) if opt.save_conf else (p.name,names[int(cls)], *xxyy,gn[1],gn[0])  # label format
                            with open(txt_path + '.csv', 'a') as f:
                                f.write(('%s, '+'%s, '+('%g, ' * (len(line)-3)+'%g').rstrip()) % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
            elif len(det)==0:
                    if save_txt:  # Write to file
                        line=(p.name, 'NULL',0,0,0,0,gn[1],gn[0],0) if opt.save_conf else (p.name, 'NULL',0,0,0,0,gn[1],gn[0])  # label format

                        with open(txt_path + '.csv', 'a') as f:
#                           
                            f.write(('%s, '+'%s, '+('%g, ' * (len(line)-3)+'%g').rstrip()) % line + '\n')

            # Print time (inference + NMS)
            # print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1) # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

#   if save_txt or save_img:
#        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
#        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--object-size-px', type=int, default=30, help='object size (pixels)')
    parser.add_argument('--tile-x', type=int, default=608, help='tile size x (pixels)')
    parser.add_argument('--tile-y', type=int, default=1024, help='tile size y (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-img', action='store_true', help='save results to image/movie file')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--save-xxyy', action='store_false', help='true to save xywh and false to save xxyy format')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    check_requirements()
    testing=False
    
    
    if testing:
        from argparse import Namespace
#        opt=Namespace(agnostic_nms=False, augment=False, classes=None, conf_thres=0.01, device='', exist_ok=False, img_size=1024, iou_thres=0.3, name='test', project='test_test', save_conf=True, save_txt=True, source='D:\\datasets\\USGS_AerialImage_2020\\testdata20200429\\USGS_AerialImages_2019_R1_sum19_tiled\\20190517_02_S_Cam1', update=False, view_img=False, weights=[r'D:\yolo_models\USGS_AerialImages_2020\train1000\train1000_pmAP_l\weights\best.pt'], save_xxyy=True)
        #opt=Namespace(agnostic_nms=False, augment=False, classes=None, conf_thres=0.5, device='', exist_ok=False, img_size=8688, object_size_px=30, tile_x=1024, tile_y=608, iou_thres=0.3, name='test', project='test_test', save_conf=True, save_img=False,save_txt=True, source="D:/CM,Inc/Dropbox (CMI)/CMI_Team/Analysis/2019/USGS_AerialImage_2019/CVAT_test_images*", update=False, view_img=False, weights=[r'D:\yolo_models\USGS_AerialImages_2020\train1000_v2\train1000_v2_pmAP_l_mosaic_flip_loscale\weights\best.pt'], save_xxyy=False)
        #opt=Namespace(agnostic_nms=False, augment=False, classes=None, conf_thres=0.5, device='', exist_ok=False, img_size=4000, object_size_px=30, tile_x=1024, tile_y=608, iou_thres=0.3, name='test', project='test_test', save_conf=True, save_txt=True, source='D:/DJI_0648.MP4', update=False, save_img=True, view_img=False, weights=[r'D:\yolo_models\USGS_AerialImages_2020\train1000_v2\train1000_v2_pmAP_l_mosaic_flip_loscale\weights\best.pt'], save_xxyy=False)
        opt=Namespace(agnostic_nms=False, augment=False, classes=None, conf_thres=0.5, device='', exist_ok=False, img_size=10000, object_size_px=16, tile_x=500, tile_y=250, iou_thres=0.3, name='pb_big_tile', project='pb_big_tile', save_conf=True, save_txt=True, source='D:/croz_20201129_v1_tiled-3-4.tif', update=False, save_img=True, view_img=False, weights=[r'D:/CM,Inc/Dropbox (CMI)/CMI_Team/Analysis/2019/PointBlue_Penguins_2019/models/exp42/weights/best.pt'], save_xxyy=False)
    
    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
