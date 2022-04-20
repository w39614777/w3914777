#include "function.h"
#ifdef AMSTENCIL
__global__ void get_max_diff1(highprecision* con,highprecision* max_diff_con){
    int unitindex_x=threadIdx.x + blockIdx.x * blockDim.x;
    int unitindex_y=threadIdx.y + blockIdx.y * blockDim.y;
    int x_start=unitindex_x*unitx;
    int y_start=unitindex_y*unity;
    highprecision(*cond)[dimX]=(highprecision(*)[dimX])con;
    highprecision(*max_diff_cond)[unitdimX]=(highprecision(*)[unitdimX])max_diff_con;
    highprecision mincon=1.0,maxcon=0.0;
    int p[4]={0,5,10,15};
    for(int j=0;j<4;j++){
        for(int i=0;i<4;i++){
            if(cond[y_start+p[j]][x_start+p[i]]<mincon)mincon=cond[y_start+p[j]][x_start+p[i]];
            if(cond[y_start+p[j]][x_start+p[i]]>maxcon)maxcon=cond[y_start+p[j]][x_start+p[i]];
        }
    }
    max_diff_cond[unitindex_y][unitindex_x]=maxcon-mincon;
}
__global__ void get_max_diff2(highprecision* test_data_last,highprecision* test_data,highprecision* max_diffs){
    int unitindex_x=threadIdx.x + blockIdx.x * blockDim.x;
    int unitindex_y=threadIdx.y + blockIdx.y * blockDim.y;
    int x_start=unitindex_x*unitx;
    int y_start=unitindex_y*unity;
    highprecision(*test_data_lastd)[dimX]=(highprecision(*)[dimX])test_data_last;
    highprecision(*test_datad)[dimX]=(highprecision(*)[dimX])test_data;
    highprecision(*max_diffsd)[unitdimX]=(highprecision(*)[unitdimX])max_diffs;
    highprecision max_diff=0.0;
    int p[4]={2,6,10,14};
    for(int j=0;j<4;j++){
        for(int i=0;i<4;i++){
            highprecision diff_this=abs(test_data_lastd[y_start+p[j]][x_start+p[i]]-test_datad[y_start+p[j]][x_start+p[i]]);
            if(diff_this>max_diff)max_diff=diff_this;
        }
    }
    max_diffsd[unitindex_y][unitindex_x]=max_diff;
}
__global__ void get_type(highprecision* max_diff_con,int *type_old,int *type_con){
    int unitindex_x=threadIdx.x + blockIdx.x * blockDim.x;
    int unitindex_y=threadIdx.y + blockIdx.y * blockDim.y;
    highprecision(*max_diff_cond)[unitdimX]=(highprecision(*)[unitdimX])max_diff_con;
    int(*type_cond)[unitdimX]=(int(*)[unitdimX])type_con;
    int(*type_oldd)[unitdimX]=(int(*)[unitdimX])type_old;
    type_oldd[unitindex_y][unitindex_x]=type_cond[unitindex_y][unitindex_x];
    highprecision max_con=max_diff_cond[unitindex_y][unitindex_x];
    max_con=unitindex_y<unitdimY-1?max(max_con,max_diff_cond[unitindex_y+1][unitindex_x]):max_con;
    max_con=unitindex_y>0?max(max_con,max_diff_cond[unitindex_y-1][unitindex_x]):max_con;
    max_con=unitindex_x<unitdimX-1?max(max_con,max_diff_cond[unitindex_y][unitindex_x+1]):max_con;
    max_con=unitindex_x>0?max(max_con,max_diff_cond[unitindex_y][unitindex_x-1]):max_con;
	if(max_con<threshold){
        type_cond[unitindex_y][unitindex_x]=1;
    }
	else if(max_con>=threshold){
        type_cond[unitindex_y][unitindex_x]=2;
    }
}
__global__ void data_sychro_duringcomputation(highprecision* con,highprecision* eta1,highprecision* eta2,half2* hcon,half2* heta1,half2* heta2,int* old_con,int* new_con){
    int unitindex_x=blockIdx.z%unitdimX;
    int unitindex_y=blockIdx.z/unitdimX;
    int(*old_cond)[unitdimX]=(int(*)[unitdimX])old_con;
    int(*new_cond)[unitdimX]=(int(*)[unitdimX])new_con;
    int x_offset=threadIdx.x ;
    int x_start=unitindex_x*uxd2;
    int x=x_start+x_offset;
    int y=unitindex_y*unity+threadIdx.y;
    if(old_cond[unitindex_y][unitindex_x]==2&&new_cond[unitindex_y][unitindex_x]==1){
        highprecision(*cond)[dimX]=(highprecision(*)[dimX])con;
        highprecision(*eta1d)[dimX]=(highprecision(*)[dimX])eta1;
        highprecision(*eta2d)[dimX]=(highprecision(*)[dimX])eta2;
        half2(*hcond)[dimXd2]=(half2(*)[dimXd2])hcon;
        half2(*heta1d)[dimXd2]=(half2(*)[dimXd2])heta1;
        half2(*heta2d)[dimXd2]=(half2(*)[dimXd2])heta2;
        hcond[y][x]=__floats2half2_rn(cond[y][x_start*2+x_offset],cond[y][x_start*2+x_offset+uxd2]);
        heta1d[y][x]=__floats2half2_rn(eta1d[y][x_start*2+x_offset],eta1d[y][x_start*2+x_offset+uxd2]);
        heta2d[y][x]=__floats2half2_rn(eta2d[y][x_start*2+x_offset],eta2d[y][x_start*2+x_offset+uxd2]);
    }
}
#endif
#if ((defined AMSTENCIL)||(defined GRAM1||(defined GRAM2)))
__global__ void dataprepare(highprecision *highdata,lowprecision *lowdata){
    highprecision(*highdatad)[dimX]=(highprecision(*)[dimX])highdata;
    lowprecision(*lowdatad)[lowprecison_dimX]=(lowprecision(*)[lowprecison_dimX])lowdata;
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    #ifdef AMSTENCIL
    lowdatad[y][x]=__floats2half2_rn(highdatad[y][x/uxd2*unitx+x%uxd2],highdatad[y][x/uxd2*unitx+x%uxd2+uxd2]);
    #endif
    #if ((defined GRAM1 )|| (defined GRAM2))
    lowdatad[y][x/uxd2*unitx+x%uxd2]=highdatad[y][x/uxd2*unitx+x%uxd2];
    lowdatad[y][x/uxd2*unitx+x%uxd2+uxd2]=highdatad[y][x/uxd2*unitx+x%uxd2+uxd2];
    #endif
}


__global__ void data_sychro_aftercomputation(highprecision *highdata,lowprecision *lowdata,int* typedata){
    highprecision(*highdatad)[dimX]=(highprecision(*)[dimX])highdata;
    lowprecision(*lowdatad)[lowprecison_dimX]=(lowprecision(*)[lowprecison_dimX])lowdata;
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int unitindex_x=x/uxd2;
    int unitindex_y=y/unity;
    #ifdef AMSTENCIL
    int(*typedatad)[unitdimX]=(int(*)[unitdimX])typedata;
    if(typedatad[unitindex_y][unitindex_x]==1){
        highdatad[y][unitindex_x*unitx+x%uxd2]=__low2float(lowdatad[y][x]);
        highdatad[y][unitindex_x*unitx+x%uxd2+uxd2]=__high2float(lowdatad[y][x]);
    }
    #endif
    #ifdef GRAM1
    if((unitindex_y*unitdimX+unitindex_x)<unitNums*ratio){
        highdatad[y][unitindex_x*unitx+x%uxd2]=__half2float(lowdatad[y][unitindex_x*unitx+x%uxd2]);
        highdatad[y][unitindex_x*unitx+x%uxd2+uxd2]=__half2float(lowdatad[y][unitindex_x*unitx+x%uxd2+uxd2]);
    }
    #endif
    #ifdef GRAM2
    if((unitindex_y*unitdimX+unitindex_x)%100<100*ratio){
        highdatad[y][unitindex_x*unitx+x%uxd2]=__half2float(lowdatad[y][unitindex_x*unitx+x%uxd2]);
        highdatad[y][unitindex_x*unitx+x%uxd2+uxd2]=__half2float(lowdatad[y][unitindex_x*unitx+x%uxd2+uxd2]);
    }
    #endif
}
lowprecision lowprecision2highprecision(highprecision a){
    #ifdef AMSTENCIL
    return __floats2half2_rn(a,a);
    #endif
    #if (defined GRAM1)||(defined GRAM2)
    return __float2half(a);
    #endif
}

#endif
template<typename T>
void writetocsv(string filename,T* data,int X,int Y){
    ofstream f(filename);
    f<<",";
    for(int x=0;x<X;x++){
        if(x==X-1)f<<x<<"\n";
        else f<<x<<",";
    }
    for(int y=0;y<Y;y++){
        f<<y<<",";
        for(int x=0;x<X;x++){
            if(x==X-1){
                f<<data[y*X+x]<<"\n";
            }else{
                f<<data[y*X+x]<<",";
            }
        }
    }
    f.close();
}
template<typename T>
void BubbleSort(T *p,int length,int* ind_diff){
    for(int m=0;m<length;m++)ind_diff[m]=m;
    for(int i=0;i<length;i++){
        for(int j=0;j<length-i-1;j++){
            if(p[j]<p[j+1]){
                T temp=p[j];
                p[j]=p[j+1];
                p[j+1]=temp;
                int ind_temp=ind_diff[j];
                ind_diff[j]=ind_diff[j+1];
                ind_diff[j+1]=ind_temp;
            }
        }
    }
}
int get_neibour(int index,int direct,int r){
    int x=index%unitdimX;
    int y=index/unitdimX;
    if(direct==1)x=(x-r+unitdimX)%unitdimX;//left
    if(direct==2)x=(x+r)%unitdimX;//right
    if(direct==3)y=(y-r+unitdimY)%unitdimY;//down
    if(direct==4)y=(y+r)%unitdimY;//up
    if(direct==5){
        x=(x-r+unitdimX)%unitdimX;
        y=(y-r+unitdimY)%unitdimY;
    }
    if(direct==6){
        x=(x+r)%unitdimX;
        y=(y-r+unitdimY)%unitdimY;
    }
    if(direct==7){
        x=(x+r)%unitdimX;
        y=(y+r)%unitdimY;
    }
    if(direct==8){
        x=(x-r+unitdimX)%unitdimX;
        y=(y+r)%unitdimY;
    }
    return y*unitdimX+x;
}