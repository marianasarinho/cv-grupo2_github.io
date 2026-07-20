#!/usr/bin/env python3
from __future__ import annotations
import argparse, glob
from pathlib import Path
import cv2, numpy as np

def mat(fs,k):
    n=fs.getNode(k); m=n.mat()
    if m is None: raise KeyError(k)
    return m

def main():
    p=argparse.ArgumentParser(); p.add_argument('--calibracao',default='saida/stereo_params_abc.xml'); p.add_argument('--pasta',default='capturas'); p.add_argument('--indice',type=int,default=-1); a=p.parse_args()
    fs=cv2.FileStorage(a.calibracao,cv2.FILE_STORAGE_READ)
    if not fs.isOpened(): raise FileNotFoundError(a.calibracao)
    w=int(fs.getNode('image_width').real()); h=int(fs.getNode('image_height').real())
    M1,D1,M2,D2,R1,R2,P1,P2=[mat(fs,k) for k in ('M1','D1','M2','D2','R1','R2','P1','P2')]; fs.release()
    mx1,my1=cv2.initUndistortRectifyMap(M1,D1,R1,P1,(w,h),cv2.CV_32FC1); mx2,my2=cv2.initUndistortRectifyMap(M2,D2,R2,P2,(w,h),cv2.CV_32FC1)
    es=sorted(glob.glob(str(Path(a.pasta)/'esquerda'/'*.png'))); ds=sorted(glob.glob(str(Path(a.pasta)/'direita'/'*.png')))
    idx=a.indice if a.indice>=0 else len(es)-1
    l=cv2.imread(es[idx]); r=cv2.imread(ds[idx]); l=cv2.remap(l,mx1,my1,cv2.INTER_LINEAR); r=cv2.remap(r,mx2,my2,cv2.INTER_LINEAR)
    par=np.hstack([l,r])
    for y in range(20,h,40): cv2.line(par,(0,y),(2*w,y),(0,255,0),1)
    cv2.imshow('Par retificado - confira mesma altura',par); cv2.imwrite('saida/verificacao_retificacao.png',par)
    print('Verifique se os mesmos pontos aparecem na mesma linha verde. Pressione qualquer tecla.')
    cv2.waitKey(0); cv2.destroyAllWindows()
if __name__=='__main__': main()
