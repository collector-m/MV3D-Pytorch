from net.common import *

def imshow(name, image, resize=1):
    H,W = image.shape[0:2]
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image.astype(np.uint8))
    cv2.resizeWindow(name, round(resize*W), round(resize*H))




def draw_shadow_text(img, text, pt,  fontScale, color, thickness, color1=None, thickness1=None):

    if color1 is None: color1=(0,0,0)
    if thickness1 is None: thickness1 = thickness+2

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, pt, font, fontScale, color1, thickness1, cv2.LINE_AA)
    cv2.putText(img, text, pt, font, fontScale, color,  thickness,  cv2.LINE_AA)



def dir_to_avi(avi_file, png_dir):

    tmp_dir = '~temp_png'
    os.makedirs(tmp_dir, exist_ok=True)

    for i, file in enumerate(sorted(glob.glob(png_dir + '/*.png'))):
        name = os.path.basename(file).replace('.png','')
        ##os.system('cp file %s'%(tmp_dir + '/' + '%06'%i + '.png'))

        png_file = png_dir +'/'+name+'.png'
        tmp_file = tmp_dir + '/%06d.png'%i
        img = cv2.imread(png_file,1)
        draw_shadow_text(img, 'timestamp='+name.replace('_',':'), (5,20),  0.5, (225,225,225), 1)
        imshow('img',img)
        cv2.waitKey(1)
        cv2.imwrite(tmp_file,img)


    os.system('ffmpeg -y -loglevel 0 -f image2 -r 15 -i %s/%%06d.png -b:v 8000k %s'%(tmp_dir,avi_file))
    os.system('rm -rf %s'%tmp_dir)

