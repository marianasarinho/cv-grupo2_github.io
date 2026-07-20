#!/usr/bin/env python3
from __future__ import annotations
import argparse, glob
from pathlib import Path
import cv2, numpy as np

def salvar_xml(path, size, square, M1,D1,M2,D2,R,T,E,F,R1,R2,P1,P2,Q,rms1,rms2,rmsS):
    fs=cv2.FileStorage(str(path),cv2.FILE_STORAGE_WRITE)
    fs.write('image_width',size[0]); fs.write('image_height',size[1]); fs.write('square_size',float(square))
    for k,v in [('M1',M1),('D1',D1),('M2',M2),('D2',D2),('R',R),('T',T),('E',E),('F',F),('R1',R1),('R2',R2),('P1',P1),('P2',P2),('Q',Q)]: fs.write(k,v)
    fs.write('rms_left',float(rms1)); fs.write('rms_right',float(rms2)); fs.write('rms_stereo',float(rmsS)); fs.release()

def main():
    p=argparse.ArgumentParser(description='Calibra duas cameras e o sistema estereo a partir dos pares.')
    p.add_argument('--pasta',default='capturas'); p.add_argument('--colunas',type=int,default=9); p.add_argument('--linhas',type=int,default=6)
    p.add_argument('--quadrado-cm',type=float,default=3.0); p.add_argument('--saida',default='saida/stereo_params_abc.xml')
    p.add_argument('--alpha',type=float,default=0.0,help='0 corta bordas; 1 preserva todo campo')
    a=p.parse_args()
    es=sorted(glob.glob(str(Path(a.pasta)/'esquerda'/'*.png'))); ds=sorted(glob.glob(str(Path(a.pasta)/'direita'/'*.png')))
    if len(es)!=len(ds) or len(es)<10: raise RuntimeError(f'Sao necessarios pelo menos 10 pares. Encontrados: E={len(es)}, D={len(ds)}')
    obj=np.zeros((a.linhas*a.colunas,3),np.float32); obj[:,:2]=np.mgrid[0:a.colunas,0:a.linhas].T.reshape(-1,2); obj*=a.quadrado_cm
    op=[]; ip1=[]; ip2=[]; usados=[]; size=None
    criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,100,1e-5)
    flags=cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_NORMALIZE_IMAGE
    for i,(le,ld) in enumerate(zip(es,ds),1):
        im1=cv2.imread(le); im2=cv2.imread(ld)
        if im1 is None or im2 is None: continue
        if im1.shape[:2]!=im2.shape[:2]: continue
        g1=cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY); g2=cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY); size=(g1.shape[1],g1.shape[0])
        r1,c1=cv2.findChessboardCorners(g1,(a.colunas,a.linhas),flags); r2,c2=cv2.findChessboardCorners(g2,(a.colunas,a.linhas),flags)
        if r1 and r2:
            c1=cv2.cornerSubPix(g1,c1,(11,11),(-1,-1),criteria); c2=cv2.cornerSubPix(g2,c2,(11,11),(-1,-1),criteria)
            op.append(obj.copy()); ip1.append(c1); ip2.append(c2); usados.append(i)
    if len(op)<10: raise RuntimeError(f'Apenas {len(op)} pares validos. Capture mais imagens variadas.')
    rms1,M1,D1,_,_=cv2.calibrateCamera(op,ip1,size,None,None)
    rms2,M2,D2,_,_=cv2.calibrateCamera(op,ip2,size,None,None)
    stereo_flags=cv2.CALIB_FIX_INTRINSIC
    rmsS,M1,D1,M2,D2,R,T,E,F=cv2.stereoCalibrate(op,ip1,ip2,M1,D1,M2,D2,size,criteria=criteria,flags=stereo_flags)
    R1,R2,P1,P2,Q,roi1,roi2=cv2.stereoRectify(M1,D1,M2,D2,size,R,T,flags=cv2.CALIB_ZERO_DISPARITY,alpha=a.alpha)
    out=Path(a.saida); out.parent.mkdir(parents=True,exist_ok=True)
    salvar_xml(out,size,a.quadrado_cm,M1,D1,M2,D2,R,T,E,F,R1,R2,P1,P2,Q,rms1,rms2,rmsS)
    print(f'Pares validos: {len(op)} de {len(es)} | indices: {usados}')
    print(f'RMS esquerda: {rms1:.4f} px | RMS direita: {rms2:.4f} px | RMS stereo: {rmsS:.4f} px')
    print(f'Baseline estimada: {np.linalg.norm(T):.3f} cm')
    print(f'Arquivo salvo em: {out.resolve()}')
if __name__=='__main__': main()
