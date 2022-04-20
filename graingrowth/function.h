#ifndef FUNCTION_H
    #define FUNCTION_H
    #include"./paras/para3.h"
    #if ((defined PURE)||(defined Motivation))
        __global__ void kernel1_pure(highprecision *etaa,highprecision *etab,highprecision* etaa_lap,highprecision *dfdetaa,highprecision *etaa_out){
            int unitindex_x=blockIdx.z%unitdimX;
            int unitindex_y=blockIdx.z/unitdimX;
            int x_offset=threadIdx.x ;
            int x_start=unitindex_x*unitx;
            int x=x_start+x_offset;
            int y_offset=threadIdx.y;
            int y=unitindex_y*unity+y_offset;
            int xs1=x>0?x-1:dimX-1;int ys1=y>0?y-1:dimY-1;
            int xa1=x<dimX-1?x+1:0;int ya1=y<dimY-1?y+1:0;
            highprecision(*etaad)[dimX]=(highprecision(*)[dimX])etaa;
            highprecision(*etabd)[dimX]=(highprecision(*)[dimX])etab;
            highprecision(*etaa_lapd)[dimX]=(highprecision(*)[dimX])etaa_lap;
            highprecision(*dfdetaad)[dimX]=(highprecision(*)[dimX])dfdetaa;
            highprecision(*etaa_outd)[dimX]=(highprecision(*)[dimX])etaa_out;
            etaa_lapd[y][x]=(etaad[ys1][xs1]+etaad[ys1][x]+etaad[ys1][xa1]+etaad[y][xs1]+etaad[y][xa1]+etaad[ya1][xs1]+etaad[ya1][x]+etaad[ya1][xa1]-8.0*etaad[y][x])/dxdy;
            highprecision sum=pow(etabd[y][x],2);
            dfdetaad[y][x]=1.0*(2.0*1.0*etaad[y][x]*sum+pow(etaad[y][x],3)-etaad[y][x]);
            etaa_outd[y][x]=etaad[y][x]-dtime*mobil*(dfdetaad[y][x]-grcoef*etaa_lapd[y][x]);
            #ifdef PRINT_INFO
            // if(x==testx&&y==testy){
            //     printf("eta center:%f,xs1:%f,xa1:%f,ys1:%f,ya1:%f\n",etaad[y][x],etaad[y][xs1],etaad[y][xa1],etaad[ys1][x],etaad[ya1][x]);
            //     printf("etaa:lap:%f,dfdetaa:%f,etaaout:%f\n",etaa_lapd[y][x],dfdetaad[y][x],etaa_outd[y][x]);
            // }
            #endif
        }
    #endif
    #if ((defined AMSTENCIL)||(defined GRAM1||(defined GRAM2)))
        __global__ void kernel1_mix(highprecision *etaa,highprecision *etaa_last,highprecision *etab,highprecision* etaa_lap,highprecision *dfdetaa,highprecision *etaa_out,lowprecision *hetaa,lowprecision *hetab,lowprecision* hetaa_lap,lowprecision *hdfdetaa,lowprecision *hetaa_out,lowprecision height,lowprecision hdxdy,lowprecision hone,lowprecision htwo,lowprecision hdtime,lowprecision hmobil,lowprecision hgrcoef,int *type_curr,int i,int etai){
            int unitindex_x=blockIdx.z%unitdimX;
            int unitindex_y=blockIdx.z/unitdimX;
            int type;
            #ifdef AMSTENCIL
                int(*type_currd)[unitdimX]=(int(*)[unitdimX])type_curr;
                type=type_currd[unitindex_y][unitindex_x];
                if(type==1&&blockIdx.x==1)return;
            #endif
            #ifdef GRAM1
                if(blockIdx.z<unitNums*ratio)type=1;
                else type=2;
            #endif
            #ifdef GRAM2
                if(blockIdx.z%100<100*ratio)type=1;
                else type=2;
            #endif
            int unitindex_ys1=unitindex_y==0?unitdimY-1:unitindex_y-1;
            int unitindex_ya1=unitindex_y==unitdimY-1?0:unitindex_y+1;
            int x_offset=threadIdx.x;
            int x_start=unitindex_x*uxd2;
            int y_offset=threadIdx.y;
            int y=unitindex_y*unity+y_offset;
            int ys1=y>0?y-1:dimY-1;
            int ya1=y<dimY-1?y+1:0;
            highprecision(*etaad)[dimX]=(highprecision(*)[dimX])etaa;
            highprecision(*etabd)[dimX]=(highprecision(*)[dimX])etab;
            highprecision(*etaa_lapd)[dimX]=(highprecision(*)[dimX])etaa_lap;
            highprecision(*dfdetaad)[dimX]=(highprecision(*)[dimX])dfdetaa;
            highprecision(*etaa_outd)[dimX]=(highprecision(*)[dimX])etaa_out;
            lowprecision(*hetaad)[lowprecison_dimX]=(lowprecision(*)[lowprecison_dimX])hetaa;
            lowprecision(*hetabd)[lowprecison_dimX]=(lowprecision(*)[lowprecison_dimX])hetab;
            lowprecision(*hetaa_lapd)[lowprecison_dimX]=(lowprecision(*)[lowprecison_dimX])hetaa_lap;
            lowprecision(*hdfdetaad)[lowprecison_dimX]=(lowprecision(*)[lowprecison_dimX])hdfdetaa;
            lowprecision(*hetaa_outd)[lowprecison_dimX]=(lowprecision(*)[lowprecison_dimX])hetaa_out;
            #if ((defined End2end)&&(defined Monitor2))
                if(i%10==0 && etai==2){
                    highprecision(*etaa_lastd)[dimX]=(highprecision(*)[dimX])etaa_last;
                    etaa_lastd[y][x_start*2+x_offset+uxd2*blockIdx.x]=etaad[y][x_start*2+x_offset+uxd2*blockIdx.x];
                }
            #endif
            if(type==1){
                int unitindex_xs1=unitindex_x==0?unitdimX-1:unitindex_x-1;
                int unitindex_xa1=unitindex_x==unitdimX-1?0:unitindex_x+1;
                //index计算
                #ifdef AMSTENCIL
                    int x=x_start+x_offset;
                #else
                    x_offset=x_offset+blockIdx.x*uxd2;
                    int x=x_start*2+x_offset;
                #endif
                int xs1=x>0?x-1:lowprecison_dimX-1;
                int xa1=x<lowprecison_dimX-1?x+1:0;
                lowprecision etaa_ys1_xs1,etaa_ys1_x,etaa_ys1_xa1,etaa_y_xs1,etaa_y_xa1,etaa_ya1_xs1,etaa_ya1_x,etaa_ya1_xa1;
                #ifdef AMSTENCIL
                    etaa_ys1_xs1=__halves2half2(__high2half(hetaad[ys1][xs1]),__low2half(hetaad[ys1][x]));
                    etaa_ys1_x=hetaad[ys1][x];
                    etaa_ys1_xa1=__halves2half2(__high2half(hetaad[ys1][x]),__low2half(hetaad[ys1][xa1]));
                    etaa_y_xs1=__halves2half2(__high2half(hetaad[y][xs1]),__low2half(hetaad[y][x]));
                    etaa_y_xa1=__halves2half2(__high2half(hetaad[y][x]),__low2half(hetaad[y][xa1]));
                    etaa_ya1_xs1=__halves2half2(__high2half(hetaad[ya1][xs1]),__low2half(hetaad[ya1][x]));
                    etaa_ya1_x=hetaad[ya1][x];
                    etaa_ya1_xa1=__halves2half2(__high2half(hetaad[ya1][x]),__low2half(hetaad[ya1][xa1]));               
                    if(x_offset==0 and y_offset==0){
                        if(type_currd[unitindex_y][unitindex_xs1]==2){
                            etaa_y_xs1=__halves2half2(__float2half(etaad[y][xs1*2+1]),__high2half(etaa_y_xs1));
                            etaa_ya1_xs1=__halves2half2(__float2half(etaad[ya1][xs1*2+1]),__high2half(etaa_ya1_xs1));
                        }
                        if(type_currd[unitindex_ys1][unitindex_x]==2){
                            etaa_ys1_xs1=__halves2half2(__low2half(etaa_ys1_xs1),__float2half(etaad[ys1][x*2]));
                            etaa_ys1_x=__floats2half2_rn(etaad[ys1][x*2],etaad[ys1][x*2+1]);
                            etaa_ys1_xa1=__floats2half2_rn(etaad[ys1][x*2+1],etaad[ys1][xa1*2]);
                        }
                        if(type_currd[unitindex_ys1][unitindex_xs1]==2){
                            etaa_ys1_xs1=__halves2half2(__float2half(etaad[ys1][xs1*2+1]),__high2half(etaa_ys1_xs1));
                        }
                    }
                    else if(x_offset==0 and y_offset==uys1){
                        if(type_currd[unitindex_y][unitindex_xs1]==2){
                            etaa_y_xs1=__halves2half2(__float2half(etaad[y][xs1*2+1]),__high2half(etaa_y_xs1));
                            etaa_ys1_xs1=__halves2half2(__float2half(etaad[ys1][xs1*2+1]),__high2half(etaa_ys1_xs1));
                        }
                        if(type_currd[unitindex_ya1][unitindex_x]==2){
                            etaa_ya1_xs1=__halves2half2(__low2half(etaa_ya1_xs1),__float2half(etaad[ya1][x*2]));
                            etaa_ya1_x=__floats2half2_rn(etaad[ya1][x*2],etaad[ya1][x*2+1]);
                            etaa_ya1_xa1=__floats2half2_rn(etaad[ya1][x*2+1],etaad[ya1][xa1*2]);
                        }
                        if(type_currd[unitindex_ya1][unitindex_xs1]==2){
                            etaa_ya1_xs1=__halves2half2(__float2half(etaad[ya1][xs1*2+1]),__high2half(etaa_ya1_xs1));
                        }
                    }
                    else if(x_offset==uxd2s1 and y_offset==0){
                        if(type_currd[unitindex_y][unitindex_xa1]==2){
                            etaa_y_xa1=__halves2half2(__low2half(etaa_y_xa1),__float2half(etaad[y][xa1*2]));
                            etaa_ya1_xa1=__halves2half2(__low2half(etaa_ya1_xa1),__float2half(etaad[ya1][xa1*2]));
                        }
                        if(type_currd[unitindex_ys1][unitindex_x]==2){
                            etaa_ys1_xs1=__floats2half2_rn(etaad[ys1][xs1*2+1],etaad[ys1][x*2]);
                            etaa_ys1_x=__floats2half2_rn(etaad[ys1][x*2],etaad[ys1][x*2+1]);
                            etaa_ys1_xa1=__halves2half2(__float2half(etaad[ys1][x*2+1]),__high2half(etaa_ys1_xa1));
                        }
                        if(type_currd[unitindex_ys1][unitindex_xa1]==2){
                            etaa_ys1_xa1=__halves2half2(__low2half(etaa_ys1_xa1),__float2half(etaad[ys1][xa1*2]));
                        }
                    }
                    else if(x_offset==uxd2s1 and y_offset==uys1){
                        if(type_currd[unitindex_y][unitindex_xa1]==2){
                            etaa_ys1_xa1=__halves2half2(__low2half(etaa_ys1_xa1),__float2half(etaad[ys1][xa1*2]));
                            etaa_y_xa1=__halves2half2(__low2half(etaa_y_xa1),__float2half(etaad[y][xa1*2]));
                        }
                        if(type_currd[unitindex_ya1][unitindex_x]==2){
                            etaa_ya1_xs1=__floats2half2_rn(etaad[ya1][xs1*2+1],etaad[ya1][x*2]);
                            etaa_ya1_x=__floats2half2_rn(etaad[ya1][x*2],etaad[ya1][x*2+1]);
                            etaa_ya1_xa1=__halves2half2(__float2half(etaad[ya1][x*2+1]),__high2half(etaa_ya1_xa1));
                        }
                        if(type_currd[unitindex_ya1][unitindex_xa1]==2){
                            etaa_ya1_xa1=__halves2half2(__low2half(etaa_ya1_xa1),__float2half(etaad[ya1][xa1*2]));
                        }
                    }
                    else if(x_offset==0 and type_currd[unitindex_y][unitindex_xs1]==2){
                        etaa_ys1_xs1=__halves2half2(__float2half(etaad[ys1][xs1*2+1]),__high2half(etaa_ys1_xs1));
                        etaa_y_xs1=__halves2half2(__float2half(etaad[y][xs1*2+1]),__high2half(etaa_y_xs1));
                        etaa_ya1_xs1=__halves2half2(__float2half(etaad[ya1][xs1*2+1]),__high2half(etaa_ya1_xs1));
                    }
                    else if(x_offset==uxd2s1 and type_currd[unitindex_y][unitindex_xa1]==2){
                        etaa_ys1_xa1=__halves2half2(__low2half(etaa_ys1_xa1),__float2half(etaad[ys1][xa1*2]));
                        etaa_y_xa1=__halves2half2(__low2half(etaa_y_xa1),__float2half(etaad[y][xa1*2]));
                        etaa_ya1_xa1=__halves2half2(__low2half(etaa_ya1_xa1),__float2half(etaad[ya1][xa1*2]));
                    }
                    else if(y_offset==0 and type_currd[unitindex_ys1][unitindex_x]==2){
                        etaa_ys1_xs1=__floats2half2_rn(etaad[ys1][xs1*2+1],etaad[ys1][x*2]);
                        etaa_ys1_x=__floats2half2_rn(etaad[ys1][x*2],etaad[ys1][x*2+1]);
                        etaa_ys1_xa1=__floats2half2_rn(etaad[ys1][x*2+1],etaad[ys1][xa1*2]);
                    }
                    else if(y_offset==uys1 and type_currd[unitindex_ya1][unitindex_x]==2){
                        etaa_ya1_xs1=__floats2half2_rn(etaad[ya1][xs1*2+1],etaad[ya1][x*2]);
                        etaa_ya1_x=__floats2half2_rn(etaad[ya1][x*2],etaad[ya1][x*2+1]);
                        etaa_ya1_xa1=__floats2half2_rn(etaad[ya1][x*2+1],etaad[ya1][xa1*2]);
                    }
                #endif

                #ifdef GRAM1
                    etaa_ys1_xs1=((x_offset==0&&y_offset==0&&(unitindex_ys1*unitdimX+unitindex_xs1)>=unitNums*ratio)||(x_offset==0&&y_offset!=0&&(unitindex_y*unitdimX+unitindex_xs1)>=unitNums*ratio)||(y_offset==0&&x_offset!=0&&(unitindex_ys1*unitdimX+unitindex_x)>=unitNums*ratio))?__float2half(etaad[ys1][xs1]):hetaad[ys1][xs1];
                    etaa_ys1_x=(y_offset==0&&(unitindex_ys1*unitdimX+unitindex_x)>=unitNums*ratio)?__float2half(etaad[ys1][x]):hetaad[ys1][x];
                    etaa_ys1_xa1=((x_offset==uxs1&&y_offset==0&&(unitindex_ys1*unitdimX+unitindex_xa1)>=unitNums*ratio)||(x_offset==uxs1&&y_offset!=0&&(unitindex_y*unitdimX+unitindex_xa1)>=unitNums*ratio)||(y_offset==0&&x_offset!=uxs1&&(unitindex_ys1*unitdimX+unitindex_x)>=unitNums*ratio))?__float2half(etaad[ys1][xa1]):hetaad[ys1][xa1];

                    etaa_y_xs1=(x_offset==0&&(unitindex_y*unitdimX+unitindex_xs1)>=unitNums*ratio)?__float2half(etaad[y][xs1]):hetaad[y][xs1];
                    etaa_y_xa1=(x_offset==uxs1&&(unitindex_y*unitdimX+unitindex_xa1)>=unitNums*ratio)?__float2half(etaad[y][xa1]):hetaad[y][xa1];

                    etaa_ya1_xs1=((x_offset==0&&y_offset==uys1&&(unitindex_ya1*unitdimX+unitindex_xs1)>=unitNums*ratio)||(x_offset==0&&y_offset!=uys1&&(unitindex_y*unitdimX+unitindex_xs1)>=unitNums*ratio)||(y_offset==uys1&&x_offset!=0&&(unitindex_ya1*unitdimX+unitindex_x)>=unitNums*ratio))?__float2half(etaad[ya1][xs1]):hetaad[ya1][xs1];
                    etaa_ya1_x=(y_offset==uys1&&(unitindex_ya1*unitdimX+unitindex_x)>=unitNums*ratio)?__float2half(etaad[ya1][x]):hetaad[ya1][x];
                    etaa_ya1_xa1=((x_offset==uxs1&&y_offset==uys1&&(unitindex_ya1*unitdimX+unitindex_xa1)>=unitNums*ratio)||(x_offset==uxs1&&y_offset!=uys1&&(unitindex_y*unitdimX+unitindex_xa1)>=unitNums*ratio)||(y_offset==uys1&&x_offset!=uxs1&&(unitindex_ya1*unitdimX+unitindex_x)>=unitNums*ratio))?__float2half(etaad[ya1][xa1]):hetaad[ya1][xa1];
                #endif

                #ifdef GRAM2
                    etaa_ys1_xs1=((x_offset==0&&y_offset==0&&(unitindex_ys1*unitdimX+unitindex_xs1)%100>=100*ratio)||(x_offset==0&&y_offset!=0&&(unitindex_y*unitdimX+unitindex_xs1)%100>=100*ratio)||(y_offset==0&&x_offset!=0&&(unitindex_ys1*unitdimX+unitindex_x)%100>=100*ratio))?__float2half(etaad[ys1][xs1]):hetaad[ys1][xs1];
                    etaa_ys1_x=(y_offset==0&&(unitindex_ys1*unitdimX+unitindex_x)%100>=100*ratio)?__float2half(etaad[ys1][x]):hetaad[ys1][x];
                    etaa_ys1_xa1=((x_offset==uxs1&&y_offset==0&&(unitindex_ys1*unitdimX+unitindex_xa1)%100>=100*ratio)||(x_offset==uxs1&&y_offset!=0&&(unitindex_y*unitdimX+unitindex_xa1)%100>=100*ratio)||(y_offset==0&&x_offset!=uxs1&&(unitindex_ys1*unitdimX+unitindex_x)%100>=100*ratio))?__float2half(etaad[ys1][xa1]):hetaad[ys1][xa1];

                    etaa_y_xs1=(x_offset==0&&(unitindex_y*unitdimX+unitindex_xs1)%100>=100*ratio)?__float2half(etaad[y][xs1]):hetaad[y][xs1];
                    etaa_y_xa1=(x_offset==uxs1&&(unitindex_y*unitdimX+unitindex_xa1)%100>=100*ratio)?__float2half(etaad[y][xa1]):hetaad[y][xa1];

                    etaa_ya1_xs1=((x_offset==0&&y_offset==uys1&&(unitindex_ya1*unitdimX+unitindex_xs1)%100>=100*ratio)||(x_offset==0&&y_offset!=uys1&&(unitindex_y*unitdimX+unitindex_xs1)%100>=100*ratio)||(y_offset==uys1&&x_offset!=0&&(unitindex_ya1*unitdimX+unitindex_x)%100>=100*ratio))?__float2half(etaad[ya1][xs1]):hetaad[ya1][xs1];
                    etaa_ya1_x=(y_offset==uys1&&(unitindex_ya1*unitdimX+unitindex_x)%100>=100*ratio)?__float2half(etaad[ya1][x]):hetaad[ya1][x];
                    etaa_ya1_xa1=((x_offset==uxs1&&y_offset==uys1&&(unitindex_ya1*unitdimX+unitindex_xa1)%100>=100*ratio)||(x_offset==uxs1&&y_offset!=uys1&&(unitindex_y*unitdimX+unitindex_xa1)%100>=100*ratio)||(y_offset==uys1&&x_offset!=uxs1&&(unitindex_ya1*unitdimX+unitindex_x)%100>=100*ratio))?__float2half(etaad[ya1][xa1]):hetaad[ya1][xa1]; 
                #endif
                hetaa_lapd[y][x]=(etaa_ys1_xs1+etaa_ys1_x+etaa_ys1_xa1+etaa_y_xs1+etaa_y_xa1+etaa_ya1_xs1+etaa_ya1_x+etaa_ya1_xa1-height*hetaad[y][x])/hdxdy;
                lowprecision sum=hetabd[y][x]*hetabd[y][x];
                hdfdetaad[y][x]=hone*(htwo*hone*hetaad[y][x]*sum+(hetaad[y][x]*hetaad[y][x]*hetaad[y][x])-hetaad[y][x]);
                hetaa_outd[y][x]=hetaad[y][x]-hdtime*hmobil*(hdfdetaad[y][x]-hgrcoef*hetaa_lapd[y][x]);
                #ifdef monitor_conversion_dependent
                    #ifdef AMSTENCIL
                        if(y_offset==0||y_offset==uys1||(i+1)%10==0){
                            etaa_outd[y][x*2]=__low2float(hetaa_outd[y][x]);
                            etaa_outd[y][x*2+1]=__high2float(hetaa_outd[y][x]);
                        }
                        else if(x_offset==0)etaa_outd[y][x*2]=__low2float(hetaa_outd[y][x]);
                        else if(x_offset==uxd2s1)etaa_outd[y][x*2+1]=__high2float(hetaa_outd[y][x]);
                    #endif
                #endif
                #ifdef PRINT_INFO
                    
                    #ifdef AMSTENCIL
                        if(x==257/2 and y==304){
                            printf("amstencil etaalap:%f,old_etaa:%f,dfdetaa:%f,etaout:%f\n",__high2float(hetaa_lapd[y][x]),__high2float(hetaad[y][x]),__high2float(hdfdetaad[y][x]),__high2float(hetaa_outd[y][x]));
                            printf("amst ys1xs1:%f,ys1x:%f,ys1xa1:%f,yxs1:%f,yx:%f,yxa1:%f,ya1xs1:%f,ya1x:%f,ya1xa1:%f\n",__high2float(etaa_ys1_xs1),__high2float(etaa_ys1_x),__high2float(etaa_ys1_xa1),__high2float(etaa_y_xs1),__high2float(hetaad[y][x]),__high2float(etaa_y_xa1),__high2float(etaa_ya1_xs1),__high2float(etaa_ya1_x),__high2float(etaa_ya1_xa1));
                        }
                    #else
                        if(x==257 and y==304){
                            printf("gram etaalap:%f,old_etaa:%f,dfdetaa:%f,etaout:%f\n",(float)hetaa_lapd[y][x],(float)hetaad[y][x],(float)hdfdetaad[y][x],(float)hetaa_outd[y][x]);
                            printf("gram ys1xs1:%f,ys1x:%f,ys1xa1:%f,yxs1:%f,yx:%f,yxa1:%f,ya1xs1:%f,ya1x:%f,ya1xa1:%f\n",(float)etaa_ys1_xs1,(float)etaa_ys1_x,(float)etaa_ys1_xa1,(float)etaa_y_xs1,(float)hetaad[y][x],(float)etaa_y_xa1,(float)etaa_ya1_xs1,(float)etaa_ya1_x,(float)etaa_ya1_xa1);
                        }
                    #endif
                    
                #endif
            }
            else{
                x_offset=x_offset+blockIdx.x*uxd2;
                int x=x_start*2+x_offset;
                int xs1=x>0?x-1:dimX-1;
                int xa1=x<dimX-1?x+1:0;
                #ifdef AMSTENCIL
                    etaa_lapd[y][x]=(etaad[ys1][xs1]+etaad[ys1][x]+etaad[ys1][xa1]+etaad[y][xs1]+etaad[y][xa1]+etaad[ya1][xs1]+etaad[ya1][x]+etaad[ya1][xa1]-8.0*etaad[y][x])/dxdy;
                #else
                    int unitindex_xs1=unitindex_x==0?unitdimX-1:unitindex_x-1;
                    int unitindex_xa1=unitindex_x==unitdimX-1?0:unitindex_x+1;
                    highprecision etaa_ys1_xs1,etaa_ys1_x,etaa_ys1_xa1,etaa_y_xs1,etaa_y_xa1,etaa_ya1_xs1,etaa_ya1_x,etaa_ya1_xa1;
                #endif
                #ifdef GRAM1
                    etaa_ys1_xs1=((x_offset==0&&y_offset==0&&(unitindex_ys1*unitdimX+unitindex_xs1)<unitNums*ratio)||(x_offset==0&&y_offset!=0&&(unitindex_y*unitdimX+unitindex_xs1)<unitNums*ratio)||(y_offset==0&&x_offset!=0&&(unitindex_ys1*unitdimX+unitindex_x)<unitNums*ratio))?__half2float(hetaad[ys1][xs1]):etaad[ys1][xs1];
                    etaa_ys1_x=(y_offset==0&&(unitindex_ys1*unitdimX+unitindex_x)<unitNums*ratio)?__half2float(hetaad[ys1][x]):etaad[ys1][x];
                    etaa_ys1_xa1=((x_offset==uxs1&&y_offset==0&&(unitindex_ys1*unitdimX+unitindex_xa1)<unitNums*ratio)||(x_offset==uxs1&&y_offset!=0&&(unitindex_y*unitdimX+unitindex_xa1)<unitNums*ratio)||(y_offset==0&&x_offset!=uxs1&&(unitindex_ys1*unitdimX+unitindex_x)<unitNums*ratio))?__half2float(hetaad[ys1][xa1]):etaad[ys1][xa1];

                    etaa_y_xs1=(x_offset==0&&(unitindex_y*unitdimX+unitindex_xs1)<unitNums*ratio)?__half2float(hetaad[y][xs1]):etaad[y][xs1];
                    etaa_y_xa1=(x_offset==uxs1&&(unitindex_y*unitdimX+unitindex_xa1)<unitNums*ratio)?__half2float(hetaad[y][xa1]):etaad[y][xa1];

                    etaa_ya1_xs1=((x_offset==0&&y_offset==uys1&&(unitindex_ya1*unitdimX+unitindex_xs1)<unitNums*ratio)||(x_offset==0&&y_offset!=uys1&&(unitindex_y*unitdimX+unitindex_xs1)<unitNums*ratio)||(y_offset==uys1&&x_offset!=0&&(unitindex_ya1*unitdimX+unitindex_x)<unitNums*ratio))?__half2float(hetaad[ya1][xs1]):etaad[ya1][xs1];
                    etaa_ya1_x=(y_offset==uys1&&(unitindex_ya1*unitdimX+unitindex_x)<unitNums*ratio)?__half2float(hetaad[ya1][x]):etaad[ya1][x];
                    etaa_ya1_xa1=((x_offset==uxs1&&y_offset==uys1&&(unitindex_ya1*unitdimX+unitindex_xa1)<unitNums*ratio)||(x_offset==uxs1&&y_offset!=uys1&&(unitindex_y*unitdimX+unitindex_xa1)<unitNums*ratio)||(y_offset==uys1&&x_offset!=uxs1&&(unitindex_ya1*unitdimX+unitindex_x)<unitNums*ratio))?__half2float(hetaad[ya1][xa1]):etaad[ya1][xa1];
                    etaa_lapd[y][x]=(etaa_ys1_xs1+etaa_ys1_x+etaa_ys1_xa1+etaa_y_xs1+etaa_y_xa1+etaa_ya1_xs1+etaa_ya1_x+etaa_ya1_xa1-8.0*etaad[y][x])/dxdy;
                #endif
                #ifdef GRAM2
                    etaa_ys1_xs1=((x_offset==0&&y_offset==0&&(unitindex_ys1*unitdimX+unitindex_xs1)%100<100*ratio)||(x_offset==0&&y_offset!=0&&(unitindex_y*unitdimX+unitindex_xs1)%100<100*ratio)||(y_offset==0&&x_offset!=0&&(unitindex_ys1*unitdimX+unitindex_x)%100<100*ratio))?__half2float(hetaad[ys1][xs1]):etaad[ys1][xs1];
                    etaa_ys1_x=(y_offset==0&&(unitindex_ys1*unitdimX+unitindex_x)%100<100*ratio)?__half2float(hetaad[ys1][x]):etaad[ys1][x];
                    etaa_ys1_xa1=((x_offset==uxs1&&y_offset==0&&(unitindex_ys1*unitdimX+unitindex_xa1)%100<100*ratio)||(x_offset==uxs1&&y_offset!=0&&(unitindex_y*unitdimX+unitindex_xa1)%100<100*ratio)||(y_offset==0&&x_offset!=uxs1&&(unitindex_ys1*unitdimX+unitindex_x)%100<100*ratio))?__half2float(hetaad[ys1][xa1]):etaad[ys1][xa1];

                    etaa_y_xs1=(x_offset==0&&(unitindex_y*unitdimX+unitindex_xs1)%100<100*ratio)?__half2float(hetaad[y][xs1]):etaad[y][xs1];
                    etaa_y_xa1=(x_offset==uxs1&&(unitindex_y*unitdimX+unitindex_xa1)%100<100*ratio)?__half2float(hetaad[y][xa1]):etaad[y][xa1];

                    etaa_ya1_xs1=((x_offset==0&&y_offset==uys1&&(unitindex_ya1*unitdimX+unitindex_xs1)%100<100*ratio)||(x_offset==0&&y_offset!=uys1&&(unitindex_y*unitdimX+unitindex_xs1)%100<100*ratio)||(y_offset==uys1&&x_offset!=0&&(unitindex_ya1*unitdimX+unitindex_x)%100<100*ratio))?__half2float(hetaad[ya1][xs1]):etaad[ya1][xs1];
                    etaa_ya1_x=(y_offset==uys1&&(unitindex_ya1*unitdimX+unitindex_x)%100<100*ratio)?__half2float(hetaad[ya1][x]):etaad[ya1][x];
                    etaa_ya1_xa1=((x_offset==uxs1&&y_offset==uys1&&(unitindex_ya1*unitdimX+unitindex_xa1)%100<100*ratio)||(x_offset==uxs1&&y_offset!=uys1&&(unitindex_y*unitdimX+unitindex_xa1)%100<100*ratio)||(y_offset==uys1&&x_offset!=uxs1&&(unitindex_ya1*unitdimX+unitindex_x)%100<100*ratio))?__half2float(hetaad[ya1][xa1]):etaad[ya1][xa1]; 
                    etaa_lapd[y][x]=(etaa_ys1_xs1+etaa_ys1_x+etaa_ys1_xa1+etaa_y_xs1+etaa_y_xa1+etaa_ya1_xs1+etaa_ya1_x+etaa_ya1_xa1-8.0*etaad[y][x])/dxdy;
                #endif
                highprecision sum=pow(etabd[y][x],2);
                dfdetaad[y][x]=1.0*(2.0*1.0*etaad[y][x]*sum+pow(etaad[y][x],3)-etaad[y][x]);
                etaa_outd[y][x]=etaad[y][x]-dtime*mobil*(dfdetaad[y][x]-grcoef*etaa_lapd[y][x]);
                #ifdef PRINT_INFO
                if(x==511&&y==0){
                    // printf("eta center:%f,xs1:%f,xa1:%f,ys1:%f,ya1:%f\n",etaad[y][x],etaad[y][xs1],etaad[y][xa1],etaad[ys1][x],etaad[ya1][x]);
                    printf("etaa:lap:%f,dfdetaa:%f,etaaout:%f\n",etaa_lapd[y][x],dfdetaad[y][x],etaa_outd[y][x]);
                }
                #endif
            }
        }
    #endif
    #if ((defined HALF)||(defined HALF2))
        __global__ void kernel1_lowpure(purelowprecision *etaa,purelowprecision *etab,purelowprecision* etaa_lap,purelowprecision *dfdetaa,purelowprecision *etaa_out,purelowprecision hone,purelowprecision htwo,purelowprecision height,purelowprecision hdtime,purelowprecision hmobil,purelowprecision hgrcoef,purelowprecision hdxdy){
            int unitindex_x=blockIdx.z%unitdimX;
            int unitindex_y=blockIdx.z/unitdimX;
            int x_offset=threadIdx.x ;
            int x_start;
            #ifdef HALF
                x_start=unitindex_x*unitx;
            #else
                x_start=unitindex_x*uxd2;
            #endif
            int x=x_start+x_offset;
            int y_offset=threadIdx.y;
            int y=unitindex_y*unity+y_offset;
            int xs1=x>0?x-1:purelowprecision_dimX-1;int ys1=y>0?y-1:dimY-1;
            int xa1=x<purelowprecision_dimX-1?x+1:0;int ya1=y<dimY-1?y+1:0;
            purelowprecision(*etaad)[purelowprecision_dimX]=(purelowprecision(*)[purelowprecision_dimX])etaa;
            purelowprecision(*etabd)[purelowprecision_dimX]=(purelowprecision(*)[purelowprecision_dimX])etab;
            purelowprecision(*etaa_lapd)[purelowprecision_dimX]=(purelowprecision(*)[purelowprecision_dimX])etaa_lap;
            purelowprecision(*dfdetaad)[purelowprecision_dimX]=(purelowprecision(*)[purelowprecision_dimX])dfdetaa;
            purelowprecision(*etaa_outd)[purelowprecision_dimX]=(purelowprecision(*)[purelowprecision_dimX])etaa_out;
            #ifdef HALF
                etaa_lapd[y][x]=(etaad[ys1][xs1]+etaad[ys1][x]+etaad[ys1][xa1]+etaad[y][xs1]+etaad[y][xa1]+etaad[ya1][xs1]+etaad[ya1][x]+etaad[ya1][xa1]-height*etaad[y][x])/hdxdy;
            #else
                etaa_lapd[y][x]=( __halves2half2(__high2half(etaad[ys1][xs1]),__low2half(etaad[ys1][x])) + etaad[ys1][x] + __halves2half2(__high2half(etaad[ys1][x]),__low2half(etaad[ys1][xa1])) + __halves2half2(__high2half(etaad[y][xs1]),__low2half(etaad[y][x])) + __halves2half2(__high2half(etaad[y][x]),__low2half(etaad[y][xa1])) + __halves2half2(__high2half(etaad[ya1][xs1]),__low2half(etaad[ya1][x])) + etaad[ya1][x] + __halves2half2(__high2half(etaad[ya1][x]),__low2half(etaad[ya1][xa1])) - height*etaad[y][x])/hdxdy;
            #endif
            purelowprecision sum=etabd[y][x]*etabd[y][x];
            dfdetaad[y][x]=hone*(htwo*hone*etaad[y][x]*sum+(etaad[y][x]*etaad[y][x]*etaad[y][x])-etaad[y][x]);
            etaa_outd[y][x]=etaad[y][x]-hdtime*hmobil*(dfdetaad[y][x]-hgrcoef*etaa_lapd[y][x]);
        }
    #endif

    #ifdef monitor_conversion_independent
        __global__ void monitor2_lastdata_store(highprecision *etaa,highprecision *etaa_last){
            int unitindex_x=blockIdx.z%unitdimX;
            int unitindex_y=blockIdx.z/unitdimX;
            int x_offset=threadIdx.x;
            int x_start=unitindex_x*uxd2;
            int y_offset=threadIdx.y;
            int y=unitindex_y*unity+y_offset;
            highprecision(*etaad)[dimX]=(highprecision(*)[dimX])etaa;
            highprecision(*etaa_lastd)[dimX]=(highprecision(*)[dimX])etaa_last;
            etaa_lastd[y][x_start*2+x_offset+uxd2*blockIdx.x]=etaad[y][x_start*2+x_offset+uxd2*blockIdx.x];
        }
        __global__ void kernel1_conversion(highprecision *etaa_out,lowprecision *hetaa_out,int *type_curr,int i){
            int unitindex_x=blockIdx.z%unitdimX;
            int unitindex_y=blockIdx.z/unitdimX;
            int type;
            int(*type_currd)[unitdimX]=(int(*)[unitdimX])type_curr;
            type=type_currd[unitindex_y][unitindex_x];
            if(type==1&&blockIdx.x==1)return;
            int x_offset=threadIdx.x;
            int x_start=unitindex_x*uxd2;
            int y_offset=threadIdx.y;
            int y=unitindex_y*unity+y_offset;
            highprecision(*etaa_outd)[dimX]=(highprecision(*)[dimX])etaa_out;
            lowprecision(*hetaa_outd)[lowprecison_dimX]=(lowprecision(*)[lowprecison_dimX])hetaa_out;
            if(type==1){
                int x=x_start+x_offset;
                if(y_offset==0||y_offset==uys1||(i+1)%10==0){
                    etaa_outd[y][x*2]=__low2float(hetaa_outd[y][x]);
                    etaa_outd[y][x*2+1]=__high2float(hetaa_outd[y][x]);
                }
                else if(x_offset==0)etaa_outd[y][x*2]=__low2float(hetaa_outd[y][x]);
                else if(x_offset==uxd2s1)etaa_outd[y][x*2+1]=__high2float(hetaa_outd[y][x]);
            }
        }
    #endif
#endif