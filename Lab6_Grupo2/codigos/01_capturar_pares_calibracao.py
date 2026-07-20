#!/usr/bin/env python3
from __future__ import annotations
import argparse, time
from pathlib import Path
import cv2


def abrir(indice:int, largura:int, altura:int):
    cap=cv2.VideoCapture(indice, cv2.CAP_V4L2)
    if not cap.isOpened(): cap=cv2.VideoCapture(indice)
    if not cap.isOpened(): raise RuntimeError(f'Nao foi possivel abrir a camera {indice}')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, largura); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, altura)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap

def main():
    p=argparse.ArgumentParser(description='Captura pares sincronizados para calibracao estereo.')
    p.add_argument('--cam-esq',type=int,default=0); p.add_argument('--cam-dir',type=int,default=2)
    p.add_argument('--largura',type=int,default=640); p.add_argument('--altura',type=int,default=480)
    p.add_argument('--colunas',type=int,default=9,help='Numero de cantos internos na horizontal')
    p.add_argument('--linhas',type=int,default=6,help='Numero de cantos internos na vertical')
    p.add_argument('--saida',default='capturas'); a=p.parse_args()
    base=Path(a.saida); le=base/'esquerda'; ld=base/'direita'; le.mkdir(parents=True,exist_ok=True); ld.mkdir(parents=True,exist_ok=True)
    ce,cd=abrir(a.cam_esq,a.largura,a.altura),abrir(a.cam_dir,a.largura,a.altura)
    n=len(list(le.glob('esq_*.png')))
    print('ESPACO: salva apenas quando o tabuleiro for detectado nas duas cameras | Q: sair')
    try:
        while True:
            for _ in range(2): ce.grab(); cd.grab()
            ok1,fe=ce.retrieve(); ok2,fd=cd.retrieve()
            if not(ok1 and ok2): continue
            ge=cv2.cvtColor(fe,cv2.COLOR_BGR2GRAY); gd=cv2.cvtColor(fd,cv2.COLOR_BGR2GRAY)
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_NORMALIZE_IMAGE+cv2.CALIB_CB_FAST_CHECK
            re,pe=cv2.findChessboardCorners(ge,(a.colunas,a.linhas),flags)
            rd,pd=cv2.findChessboardCorners(gd,(a.colunas,a.linhas),flags)
            ve,vd=fe.copy(),fd.copy()
            if re: cv2.drawChessboardCorners(ve,(a.colunas,a.linhas),pe,re)
            if rd: cv2.drawChessboardCorners(vd,(a.colunas,a.linhas),pd,rd)
            status='OK NAS DUAS' if re and rd else 'reposicione o tabuleiro'
            cv2.putText(ve,f'{status} | pares: {n}',(10,28),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,0) if re and rd else (0,0,255),2)
            cv2.putText(vd,f'{status} | pares: {n}',(10,28),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,0) if re and rd else (0,0,255),2)
            cv2.imshow('Camera esquerda',ve); cv2.imshow('Camera direita',vd)
            k=cv2.waitKey(1)&0xFF
            if k in (ord('q'),27): break
            if k==32:
                if re and rd:
                    n+=1; ts=time.strftime('%Y%m%d_%H%M%S')
                    cv2.imwrite(str(le/f'esq_{n:03d}_{ts}.png'),fe); cv2.imwrite(str(ld/f'dir_{n:03d}_{ts}.png'),fd)
                    print(f'Par {n:03d} salvo.')
                else: print('Nao salvo: tabuleiro nao detectado simultaneamente.')
    finally:
        ce.release(); cd.release(); cv2.destroyAllWindows()
if __name__=='__main__': main()
