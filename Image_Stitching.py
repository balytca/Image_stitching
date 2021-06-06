import cv2
import numpy as np
import sys
import math

def all(arg):
    result = 0
    for e in arg:
        if e :
            result +=1
        else:
            result -=1
            
    return result
class Image_Stitching():
    def __init__(self) :
        self.ratio=0.85
        self.min_match=4
        self.threshold = 0.0002
        self.panorama_b = 0
        #self.sift=cv2.SIFT_create()
        #self.smoothing_window_size=img1.shape[1]
    def Calibration(self,img1,img2,threshold,ratio, status = 'ok'):
        
        akaze = cv2.AKAZE_create(threshold = threshold) #threshold = 0.0001, nOctaves=5, nOctaveLayers=5
        kp1, des1 = akaze.detectAndCompute(img1, None)
        kp2, des2 = akaze.detectAndCompute(img2, None)
        
        matcher = cv2.BFMatcher()
        raw_matches = matcher.knnMatch(des1, des2, k=2)
        good_points = []
        good_matches=[]
        for m1, m2 in raw_matches:
            if m1.distance < ratio * m2.distance:
                good_points.append((m1.trainIdx, m1.queryIdx))
                good_matches.append([m1])
        
                
        
        coof = [len(raw_matches)/len(des2), len(good_matches)/len(raw_matches),len(good_matches)]
        print(" {:.4f} {:.4f}  {:5d}||{:.4f}  {:.4f}".format(coof[0],coof[1],coof[2] ,threshold, ratio))
        if  coof[2] >200:
            if coof[1] > 0.15:
                status, good_points_, good_matches_, kp1_, kp2_ = self.Calibration(img1,img2,threshold,ratio-0.1)
            else:
                status, good_points_, good_matches_, kp1_, kp2_ = self.Calibration(img1,img2,threshold+0.0001,ratio)
        elif coof[2] <= 0: 
            return 'Zero', None, None, None, None
        elif  coof[2] <4:
            if not(coof[2]) or (coof[1] > 0.15):
                status, good_points_, good_matches_, kp1_, kp2_ = self.Calibration(img1,img2,threshold-0.0001,ratio)
            elif (coof[2]) and (coof[1] < 0.15):
                status, good_points_, good_matches_, kp1_, kp2_ = self.Calibration(img1,img2,threshold,ratio+0.1)   
        
        if status == 'return':
            good_points, good_matches, kp1, kp2 = good_points_, good_matches_, kp1_, kp2_
        return 'return', good_points, good_matches, kp1, kp2  
    
    def registration(self,img1,img2):
        good_points = []
        good_matches=[]
        
        status, good_points, good_matches, kp1, kp2 =  self.Calibration(img1,img2,self.threshold,self.ratio)     
        if  status == 'Zero':
            sys.exit("Изображения не совместимы")
        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=2)
        cv2.imwrite('matching.jpg', img3)
        if len(good_points) > self.min_match:
            image1_kp = np.float32(
                [kp1[i].pt for (_, i) in good_points])
            image2_kp = np.float32(
                [kp2[i].pt for (i, _) in good_points])
            H, status = cv2.findHomography(image2_kp, image1_kp, cv2.RANSAC,5.0)
            #print(H)    
        return H

    
    def rotate_image(self,mat, angle):
            
        height, width = mat.shape[:2] # image shape has 3 dimensions
        image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape
    
        rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)
    
        # rotation calculates the cos and sin, taking absolutes of those.
        abs_cos = abs(rotation_mat[0,0]) 
        abs_sin = abs(rotation_mat[0,1])
    
        # find the new width and height bounds
        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)
    
        # subtract old image center (bringing image back to origo) and adding the new image center coordinates
        rotation_mat[0, 2] += bound_w/2 - image_center[0]
        rotation_mat[1, 2] += bound_h/2 - image_center[1]
    
        # rotate image with the new bounds and translated rotation matrix
        rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
        return rotated_mat
    def warpPerspectivePadded(self, img, dst, transf):

        src_h, src_w = img.shape[:2]
        lin_homg_pts = np.array([[0, src_w, src_w, 0], [0, 0, src_h, src_h], [1, 1, 1, 1]])
    
        trans_lin_homg_pts = transf.dot(lin_homg_pts)
        trans_lin_homg_pts /= trans_lin_homg_pts[2,:]
    
        minX = np.min(trans_lin_homg_pts[0,:])
        minY = np.min(trans_lin_homg_pts[1,:])
        maxX = np.max(trans_lin_homg_pts[0,:])
        maxY = np.max(trans_lin_homg_pts[1,:])
    
        # calculate the needed padding and create a blank image to place dst within
        dst_sz = list(dst.shape)
        pad_sz = dst_sz.copy() # to get the same number of channels
        pad_sz[0] = np.round(np.maximum(dst_sz[0], maxY) - np.minimum(0, minY)).astype(int)
        pad_sz[1] = np.round(np.maximum(dst_sz[1], maxX) - np.minimum(0, minX)).astype(int)
        dst_pad = np.zeros(pad_sz, dtype=np.uint8)
        
        anchorX, anchorY = 0, 0
        transl_transf = np.eye(3,3)
        if minX < 0: 
            anchorX = np.round(-minX).astype(int)
            transl_transf[0,2] += anchorX
        if minY < 0:
            anchorY = np.round(-minY).astype(int)
            transl_transf[1,2] += anchorY
        new_transf = transl_transf.dot(transf)
        new_transf /= new_transf[2,2]
    
        dst_pad[anchorY:anchorY+dst_sz[0], anchorX:anchorX+dst_sz[1]] = dst
    
        warped = cv2.warpPerspective(img, new_transf, (pad_sz[1],pad_sz[0]))
    
        return dst_pad, warped
    def blending(self,img1,img2):
        H = self.registration(img1,img2)
        #theta = math.atan2(H[0,1], H[0,0]) * 180 / math.pi
        #img2 = self.rotate_image(img2,theta)
        img1,img2 = self.warpPerspectivePadded(img2,img1,H)
        #H = self.registration(img1,img2)
        cv2.imwrite('panorama_test2.png', img2)
        cv2.imwrite('panorama_test1.png', img1)
        
        
        height_panorama = img1.shape[0]
        width_panorama = img1.shape[1]
        result = np.zeros((height_panorama, width_panorama, 3), dtype = np.uint8)
        #edges_result = np.zeros((height_panorama, width_panorama, 3))
                
        for i,e in enumerate(img2):
            for f,o in enumerate(e):
                if img2[i][f].all() and img1[i][f].all():
                    result[i][f]+= img1[i][f]
                    
                elif img1[i][f].all():
                    result[i][f]+= img1[i][f]
                #elif img2[i][f].all():
                else: 
                    #if (i>1)and(i<img2.shape[1]-2) and(f>1)and(f<img2.shape[1]-2):
                    try:
                        #if img2[i][f].all(): 
                        if  np.array([np.array([img2[i-1:i+1][e][f-1:f+1][o].all() 
                                               for o in range(2)]).all() for e in range(2)]).all():  
                            result[i][f]+= img2[i][f]
                            #print(i,f)
                        else:
                            result[i][f]+=  img1[i][f] 
                    except:    
                    #else:
                        result[i][f]+=  img1[i][f] +img2[i][f]
               
        if self.panorama_b:
            #row = []
            for i in range(height_panorama):
                if all(result[i, :, 0] != 0)< 0.75*width_panorama:
                    min_row = i
                else:
                    break
            for i in range(height_panorama-1,-1,-1):
                if (all(result[i, :, 0]!= 0)) < 0.75*width_panorama:
                    max_row = i
                else:
                    break
            min_col, max_col = 0,  width_panorama
        else:
            rows, cols = np.where(result[:, :, 0] != 0)
            min_row, max_row = min(rows), max(rows) 
            min_col, max_col = min(cols), max(cols) 
        final_result = result[min_row:max_row, min_col:max_col, :]
        print ('---')
        #self.Testing(img1,img2,final_result)
        return final_result, img1,img2
def Testing(img1,img2, final_result_):
    
    edges1 = cv2.Canny(img1,5,100)
    edges2 = cv2.Canny(img2,5,100)
    edges_result = edges2 + edges1 #cv2.bitwise_or(edges1, edges2)
    edges_result =  cv2.cvtColor(edges_result,cv2.COLOR_GRAY2BGRA)
    edges_result = cv2.cvtColor(edges_result,cv2.COLOR_BGRA2RGB)
    kSize =[int((img1.shape[0])// 60 + np.round(((img1.shape[0])/ 60)-((img1.shape[0])// 60))) 
            , int((img1.shape[1])// 60 + np.round(((img1.shape[1])/ 60)-((img1.shape[1])// 60)))]
    CanResult = np.zeros((int(kSize[0]),int( kSize[1] )))
    for i in range(kSize[0]):
        for j in range(kSize[1]):
            
           X1 = sum([e.count(True) for e in 
                                  [ [True if (b.all()>0) else None for b in e[i*60:(i+1)*60]] 
                                   for e in edges1[j*60:(j+1)*60]]])
           
           X2 = sum([e.count(True) for e in 
                                  [ [True if (b.all()>0) else None for b in e[i*60:(i+1)*60]] 
                                       for e in edges2[j*60:(j+1)*60]]])
           cv2.line(final_result_,(i*60,j*60),(i*60,(j+1)*60),(220,30,220)) 
           cv2.line(final_result_,(i*60,j*60),((i+1)*60,(j*60)),(220,30,220)) 
           cv2.line(final_result_,(i*60,j*60),(i*60,(j-1)*60),(220,30,220)) 
           cv2.line(final_result_,(i*60,j*60),((i-1)*60,(j*60)),(220,30,220)) 
           if X1 and X2 :
               if X1 > X2:
                   CanResult[i][j] = X2 / X1
               else:
                   CanResult[i][j] = X1 / X2
               
               cv2.putText(final_result_,f"{CanResult[i][j]:.2f}",
                           (i*60,j*60 -5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,(220,220,30),1)
           cv2.line(edges_result,(i*60,j*60),(i*60,(j+1)*60),(220,30,220)) 
           cv2.line(edges_result,(i*60,j*60),((i+1)*60,(j*60)),(220,30,220)) 
           cv2.line(edges_result,(i*60,j*60),(i*60,(j-1)*60),(220,30,220)) 
           cv2.line(edges_result,(i*60,j*60),((i-1)*60,(j*60)),(220,30,220)) 
           if X1 and X2 :
               if X1 > X2:
                   CanResult[i][j] = X2 / X1
               else:
                   CanResult[i][j] = X1 / X2
               cv2.putText(edges_result,f"{CanResult[i][j]:.2f}",
                           (i*60,j*60 -5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,(220,220,220),3)
               cv2.putText(edges_result,f"{CanResult[i][j]:.2f}",
                           (i*60,j*60 -5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,(30,30,30),1) 
               
  
    
    cv2.imwrite('edges.png', edges_result)
    cv2.imwrite('finalTest.png', final_result_)
def main(argv1,argv2):
    global img1
    global img2
    img2= cv2.imread(argv2)
    if type(argv1) == type('str'):
        img1 = cv2.imread(argv1)
    elif type(argv1) == type(img2):
        img1 = argv1 # cv2.imdecode(argv1,flags = -1)
    #print(type(argv1))
    if type(img1) == type(img2) != type(None) :
        pass
    else:
        sys.exit(f"Wrong arguments {sys.argv}")
    global final
    global theta
   
    final, img1,img2 =Image_Stitching().blending(img1,img2)
    cv2.imwrite('panorama.png', final)
    if argv2 == sys.argv[-1]:
        Testing(img1,img2,final)
   
    
    

if __name__ == '__main__':
    try: 
        if len(sys.argv)> 3:
            main(sys.argv[1],sys.argv[2])
            for i in range(3,len(sys.argv)):
                main(final,sys.argv[i])
        else:
            main(sys.argv[1],sys.argv[2])
            
    except IndexError:
        #main('triplple_1.png','triplple_2.png')
        main('panorama.png','V7.png')
        #main('test3.png','test4.png')
        print ("Please input two source images: ")
        print ("For example: python Image_Stitching.py '/Users/linrl3/Desktop/picture/p1.jpg' '/Users/linrl3/Desktop/picture/p2.jpg'")
    

