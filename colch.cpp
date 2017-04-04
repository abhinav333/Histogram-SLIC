#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core/utility.hpp"
#include <sys/time.h>
#include <sys/resource.h>
#include <pthread.h>
#include "opencv2/highgui/highgui_c.h"
#include<iostream>
#include<fstream>
#include<iterator>
#include<string>
#include<vector>
#include<sys/time.h>
#include<sys/resource.h>


typedef signed int sint;
typedef double pdat;
#define ITER_MAX 5 
#define MAX_ERROR 10000

/*
-loads images in BGR order
-OpenCV Lab ->   L-> L*255/100 ,a-> a+128 ,b-> b+128 // need to reconvert these values

*/

//(Y,X)=>(ROWS,COLUMNS) notation




using namespace std;
using namespace cv;
struct rusage res;
void *test_fun(void *dat)
{
 //declare local structure pointer
 rusage *tl=(rusage *)dat;
 int i=0;
  
 i=getrusage(RUSAGE_THREAD,tl); 
 pthread_exit(dat);


}

//pixel value return: mode0=pointer_referencing mode1=using .at  

template <class M_TYPE,class R_TYPE>
R_TYPE px_val(Mat &matrix,const int y,const int x,const int mode)
{
 R_TYPE val;
 
 if(mode==1)
 {
  val=(R_TYPE)matrix.at<M_TYPE>(y,x);
 }
 else if(mode==0)
 {
  M_TYPE *p=matrix.ptr<M_TYPE>(y);
  val=(R_TYPE)*(p+x);
 }
 return val;
}

//distance metric
pdat dist_pixel(pdat& kx,pdat& ky,pdat& kl,pdat& ka,pdat& kb,pdat& pix,pdat& piy,pdat& pil, \
                   pdat& pia,pdat& pib,pdat& factor1,pdat& factor2)
{
 pdat d_m=0,d_c=0;
 d_m=(pix-kx)*(pix-kx)+(piy-ky)*(piy-ky); //euclidean 
 d_c=(pil-kl)*(pil-kl)+(pia-ka)*(pia-ka)+(pib-kb)*(pib-kb); //euclidean
 //d_m=abs(pix-kx)+abs(piy-ky); //Manhattan
 //d_c=abs(pil-kl)+abs(pia-ka)+abs(pib-kb); //Manhattan
 
 return(2*d_m/factor1+d_c/factor2);
}

void super_pixel(Mat& image,Mat& outp,int seed_n,int iter_m)
{
    
    struct timespec ts,te;  
    

    Mat imgxyz;
    

    cvtColor(image,imgxyz,CV_BGR2Lab);
    pdat fac1=1.0,fac2=1.0;    
    //probe set
    clock_gettime(CLOCK_REALTIME,&ts);
    //
    Mat channel[3];
    split(imgxyz,channel);
    channel[0].convertTo(channel[0],CV_8U,(100.0/255.0),0); //L scaling :stored in unsigned char format
    channel[1].convertTo(channel[1],CV_8S,1,-128); //a->a-128 :stored in unsigned char format
    channel[2].convertTo(channel[2],CV_8S,1,-128); //b->b-128
   
    //access using mat.at is slower than using ptr// use ptr for realtime app
    
    int im_width=channel[0].cols;
    int im_height=channel[0].rows;
    int nseeds=seed_n;
    pdat kstep=sqrt(double(im_width*im_height)/double(nseeds));
    pdat kstep_2=round(kstep/2);
    kstep=round(kstep);
    vector<pdat>kseedx; //seed x
    vector<pdat>kseedy; //seed y
    vector<pdat>kseedl; 
    vector<pdat>kseeda;
    vector<pdat>kseedb;
    
    //initial seed position
        
    sint h_seeds=(sint)(round(sqrt(double(nseeds*im_width)/double(im_height))));
    sint v_seeds=(sint)(round(sqrt(double(nseeds*im_height)/double(im_width))));

      
    //sint h_step=(sint)(kstep+(im_width-((sint)kstep*h_seeds))/kstep);
    //sint v_step=(sint)(kstep+double(im_height-(kstep*v_seeds))/kstep);
    //cout<<"IM width:"<<im_width<<" Im height:"<<im_height<<"\n";
//	cout<<kstep<<" "<<h_seeds<<" "<<v_seeds<<" "<<px_val<unsigned char,pdat>(channel[0],50,100,0);
    
    //store seeds with reqular grid step
    int x_scan,y_scan,i_scan; //general scan variables
    pdat seedx_temp=0,seedy_temp=0;
    for(y_scan=0;y_scan<v_seeds+1;y_scan++)
    {
     for(x_scan=0;x_scan<h_seeds+1;x_scan++)
     {
      seedx_temp=kstep*double(x_scan)+kstep_2;
      seedy_temp=kstep*double(y_scan)+kstep_2;
      if(seedx_temp<=im_width && seedy_temp<=im_height)
      {
       kseedx.push_back(seedx_temp);
       kseedy.push_back(seedy_temp);
       kseedl.push_back(px_val<unsigned char,pdat>(channel[0],(int)seedy_temp,(int)seedx_temp,0));      
       kseeda.push_back(px_val<signed char,pdat>(channel[1],(int)seedy_temp,(int)seedx_temp,0));      
       kseedb.push_back(px_val<signed char,pdat>(channel[2],(int)seedy_temp,(int)seedx_temp,0));      
     }
     }
    }
    
    //modify seed number
    nseeds=kseedx.size();
    
    //perturb seeds: initial placement of seeds




///////
   //Mat output=Mat::zeros(im_height,im_width,CV_8U);     
////
    // kmeans algorithm
    long npixels=im_width*im_height;
    pdat seedx=0,seedy=0,seedl=0,seeda=0,seedb=0; //seed attributes
    pdat px=0,py=0,pl=0,pa=0,pb=0; //pixel attributes
    vector<pdat>px_label(npixels,-1);
    vector<pdat>px_distance(npixels,(pdat)npixels);
    vector<pdat>cluster_points(nseeds,0);
    vector<pdat>sig_x(nseeds,0);
    vector<pdat>sig_y(nseeds,0);
    vector<pdat>sig_l(nseeds,0);
    vector<pdat>sig_a(nseeds,0);
    vector<pdat>sig_b(nseeds,0);
    vector<pdat>maximum_lab(nseeds,1);    
    int iter=0;
    long lin_index=0;
    pdat distance=0;
    pdat diff=0;
//loop:while err>e and iter<iter_max
   while(iter<iter_m)
   {
    for(i_scan=0;i_scan<nseeds;i_scan++)
    {
     seedx=kseedx[i_scan];
     seedy=kseedy[i_scan];
     seedl=kseedl[i_scan];
     seeda=kseeda[i_scan];
     seedb=kseedb[i_scan];
    
     for(y_scan=max(0,int(seedy-(2*kstep_2)));y_scan<min(im_height,int(seedy+(2*kstep_2)));y_scan++)
     {
      for(x_scan=max(0,int(seedx-(2*kstep_2)));x_scan<min(im_width,int(seedx+(2*kstep_2)));x_scan++)
      {
         
         // test.at<unsigned char>(y_scan,x_scan)=(i_scan%2)*255;
         //compute labels and pixels
         px=x_scan;
	 py=y_scan;
	 pl=px_val<unsigned char,pdat>(channel[0],py,px,0);
	 pa=px_val<signed char,pdat>(channel[1],py,px,0);
	 pb=px_val<signed char,pdat>(channel[2],py,px,0);
         distance=dist_pixel(seedx,seedy,seedl,seeda,seedb,px,py,pl,pa,pb,fac1,fac2);
         lin_index=y_scan*im_width+x_scan;
	 if(distance<px_distance[lin_index])
	 {
	  if(px_label[lin_index]!=-1) //remove attribute contributions from current cluster 
	  {
	   sig_x[int(px_label[lin_index])]-=px;
	   sig_y[int(px_label[lin_index])]-=py;
	   sig_l[int(px_label[lin_index])]-=pl;
	   sig_a[int(px_label[lin_index])]-=pa;
	   sig_b[int(px_label[lin_index])]-=pb;
           cluster_points[int(px_label[lin_index])]--;
           diff-=(px-kseedx[int(px_label[lin_index])])*(px-kseedx[int(px_label[lin_index])]);
           diff-=(py-kseedy[int(px_label[lin_index])])*(py-kseedy[int(px_label[lin_index])]);
           diff-=(pl-kseedl[int(px_label[lin_index])])*(pl-kseedl[int(px_label[lin_index])]);
           diff-=(pa-kseeda[int(px_label[lin_index])])*(pa-kseeda[int(px_label[lin_index])]);
           diff-=(pb-kseedb[int(px_label[lin_index])])*(pb-kseedb[int(px_label[lin_index])]);
          } 
	  
          //if(distance>maximum_lab[i_scan]) {maximum_lab[i_scan]=distance;}

	  px_label[lin_index]=i_scan; //enter new seed attributes
	  cluster_points[i_scan]++;
          px_distance[lin_index]=distance;
          sig_x[i_scan]+=px;
	  sig_y[i_scan]+=py;
	  sig_l[i_scan]+=pl;
	  sig_a[i_scan]+=pa;
	  sig_b[i_scan]+=pb;
          diff+=(px-kseedx[i_scan])*(px-kseedx[i_scan]);
          diff+=(py-kseedy[i_scan])*(py-kseedy[i_scan]);
          diff+=(pl-kseedl[i_scan])*(pl-kseedl[i_scan]);
          diff+=(pa-kseeda[i_scan])*(pa-kseeda[i_scan]);
          diff+=(pb-kseedb[i_scan])*(pb-kseedb[i_scan]);
	 }
             
      }//x_scan
     }//y_scan
    }//seed_scan
    //update cluster center: simple method
    for(i_scan=0;i_scan<nseeds;i_scan++)
    {
     kseedx[i_scan]=round(sig_x[i_scan]/(cluster_points[i_scan] ? cluster_points[i_scan] : 1));
     kseedy[i_scan]=round(sig_y[i_scan]/cluster_points[i_scan]);
     kseedl[i_scan]=sig_l[i_scan]/cluster_points[i_scan];
     kseeda[i_scan]=sig_a[i_scan]/cluster_points[i_scan];
     kseedb[i_scan]=sig_b[i_scan]/cluster_points[i_scan];
    }//cluster update
   iter++;
  }//while loop 
   
   clock_gettime(CLOCK_REALTIME,&te);
//create a Mat from pixel label values
  Mat output=image.clone();
  
  pdat nh_label=-1,nv_label=-1; 
  for(y_scan=0;y_scan<im_height-1;y_scan++)
  {
   for(x_scan=0;x_scan<im_width-1;x_scan++)
   {
    lin_index=y_scan*im_width+x_scan;
    nh_label=px_label[lin_index+1];
    nv_label=px_label[lin_index+im_width];
    if((px_label[lin_index]!=nh_label) || (px_label[lin_index]!=nv_label))
    {
     output.at<Vec3b>(y_scan,x_scan)[0]=255;
     output.at<Vec3b>(y_scan,x_scan)[1]=255;
     output.at<Vec3b>(y_scan,x_scan)[2]=255;
     //output.at<unsigned char>(y_scan,x_scan)=255;
      //test.at<unsigned char>(y_scan,x_scan)=(unsigned char)(px_label[lin_index]);
    }
   }
  }
  outp=output.clone();
  cout<<"\n"<<"process completed!"<<"\n"; 
  cout<<"Optimized seed count:"<<nseeds<<"\n";
  cout<<"Objective function error:"<<diff<<"\n";
  cout<<"Iterations required:"<<iter<<"\n";
  long t_s=(ts.tv_sec*pow(10,9)+ts.tv_nsec);
  long t_e=(te.tv_sec*pow(10,9)+te.tv_nsec);
  long t_second=(t_e-t_s)/long(pow(10,9));
  long t_usec=((t_e-t_s)%long(pow(10,9)))/long(pow(10,3));
  cout<<"Time required: "<<t_second<<" sec; "<<t_usec<<" usecs"<<"\n";
 
    //show seeds on the image
//    Mat test=Mat::zeros(im_height,im_width,CV_8U);     

  //  for(i_scan=0;i_scan<nseeds;i_scan++)
    //{
     //test.at<unsigned char>(int(kseedy[i_scan]),int(kseedx[i_scan]))=255;
    //}    
    
    //Mat dil_tes;
    //Canny(test,dil_tes,50,150,3);
    //dilate(test,dil_tes,Mat(),Point(-1,-1),333,1,1);  	 
   
    
    //populate blank mat with superpixel boundaries









   /*namedWindow("original", CV_WINDOW_AUTOSIZE);
   imshow("original", image);
	
    namedWindow("CIE",CV_WINDOW_AUTOSIZE);
    imshow("CIE",channel[0]);
   
    namedWindow("seeds",CV_WINDOW_AUTOSIZE);
    imshow("seeds",output);*/
    //namedWindow("parameters",CV_WINDOW_AUTOSIZE);

}

int main(void)
{
   Mat image,output;
   image = imread("img1.png",1);
  
  namedWindow("seeds",CV_WINDOW_AUTOSIZE);

    //pdat fac1=1.0,fac2=1.0;
    int seeding=50;
    int iter_max=5;
    createTrackbar("superpixels","seeds",&seeding,500,NULL);  //approximate number
    createTrackbar("iterations","seeds",&iter_max,50,NULL);
    
    while(1)
    {    

     super_pixel(image,output,1+getTrackbarPos("superpixels","seeds"), \
                1+getTrackbarPos("iterations","seeds"));
     imshow("seeds",output);
     waitKey(0);
    }
    return 0;


}
