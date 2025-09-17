from numpy import *
from matplotlib.pyplot import *
from scipy.optimize import newton, minimize_scalar
from scipy.special import expi, wrightomega
from prettytable import PrettyTable

def L_OTOR():
    #Constants
    T0=200
    k_b=8.617e-5
    n0=1
    

    #Initial parameters
    E=1
    s=1e13
    f_s="{:.4e}".format(s)
    Am=1e-7
    A_m="{:.3e}".format(Am)
    AD=1e-8
    A_D="{:.3e}".format(AD)
    
    r=AD/Am
    c=n0*Am/AD
    bita=1
    dis=0.01
    
    
    TL_values=[]
    Temperatures=[]
    n_0=0

    #TL values intensity
    for i in range(0,40000):
        T=T0+bita*i*dis
        Temperatures.append(T)
        
        E_i=expi(-E/(k_b*T))
        exi=T*exp(-E/(k_b*T))+(E/k_b)*E_i
        z=1/c-log(c)+s*exi
        
        W=wrightomega(z)
        TL_value=(n0/c)*(s/bita)*exp(-E/(k_b*T))/(W+W**2) 
        n_0+=TL_value*dis
        TL_values.append(TL_value)

    figure()
    plot(Temperatures,TL_values, "b-")
    xlabel("Temperatures K")
    ylabel("TL Intensity")
    title("TL glow peak in L-OTOR")
    show()

    #figure_path="C:\\Users\\User\\Python_files\\Thermoluminescence Glow Curve\\TL_glow_peak(LOTOR).png"
    #savefig(figure_path)

    #Intesnity max
    Imax=max(TL_values)
    Index_Imax=TL_values.index(Imax)
    #print("Imax=",Imax)

    #Temperature max
    Tmax=Temperatures[Index_Imax]
    #print("Tmax=",Tmax)

    #Closest index of half Intesity max

    def find_closest(TL_list,target):
        closest_diff=abs(TL_list[0]-target)
        closest_index=0
        for i in range(0,len(TL_list)):
            diff=abs(TL_list[i]-target)
            if diff<closest_diff:
                closest_diff=diff
                closest_index=i
        return closest_index

    Half_Intensity=Imax/2

    #Temperatures points
    T1_index=find_closest(TL_values[:Index_Imax],Half_Intensity) 
    T1=Temperatures[T1_index]
    #print("T1=",T1)

    T2_index=find_closest(TL_values[Index_Imax:],Half_Intensity)+Index_Imax
    T2=Temperatures[T2_index]
    #print("T2=",T2)

    #nm calculation
    nm=0
    for i in range(Index_Imax,len(TL_values)):
        nm+=TL_values[i]*dis
    #print(n_0,nm)

    #Ca and a=δ,ω,τ calculation
    omega=T2-T1
    thelta=T2-Tmax
    tau=Tmax-T1
    Comega=(omega*Imax)/(bita*n_0)
    Cthelta=(thelta*Imax)/(bita*nm)
    Ctau=(tau*Imax)/(bita*(n_0-nm))
    mg=thelta/omega
    mg_tone=nm/n_0
    #print(omega,tau,thelta,Comega,Cthelta,Ctau,mg,mg_tone)

    #Calculation of r
    
    r_initial=1
    best_result=1e-5  
    mg_tone=nm/n_0
    #E=Activation_Energy()
    Dm=2*k_b*Tmax/E
    for i in range(1,1000000):
        
        r=i/100000
        z=r+log(r)+(1-Dm)*(1+0.7617*(r)**1.1486)
        Wz=wrightomega(z)
        result=abs(mg_tone-(r/Wz))
                
        if result<best_result:
            best_result=result
            r_initial=float(r)
   
    best_r=r_initial
    
    if best_r==1:
        def f_r(r):
            Dm=2*k_b*Tmax/E
            z=r+log(r)+(1-Dm)*(1+0.7617*(r)**1.1486)
            Wz=wrightomega(z)
            return abs(mg_tone-(r/Wz))

        
        result = minimize_scalar(f_r, bounds=(1e-6, 1e6), method='bounded')
        best_r = result.x
    
    # Calculation of required coefficients for activation energy
    zm=best_r+log(best_r)+(1-Dm)*(1+0.7617*(best_r)**1.1486)
    Wzm=wrightomega(zm)
    FTL=(1+0.7617*(best_r)**1.1486)
    FR=(1/(best_r*FTL))*(Wzm+Wzm**2)
    #print(FTL,FR)

    #Activation energy calculation
    
    r_root=best_r
    Eomega=(Comega*k_b*Tmax**2)*FR/omega
    Ethelta=Cthelta*mg_tone*FR*(k_b*Tmax**2/thelta)
    Etau=Ctau*(1-mg_tone)*FR*(k_b*Tmax**2/tau)
    #print(Eomega,Ethelta,Etau)

    Error=abs(E-Eomega)*100/E
    exp_part=exp(E/(k_b*Tmax))
    s_calc=bita*E*exp_part/(k_b*Tmax**2*(1+0.7617326*best_r**1.1485711))
    s1="{:.3e}".format(s_calc)
    
    Table1=PrettyTable(["E","r","s","AD","Am","bita"])
    Table1.add_row([E,AD/Am,f_s,A_D,A_m,bita])
    print("Initial conditions")
    print(Table1,'\n')

    Table2=PrettyTable(["T1","Tmax","T2"])
    Table2.add_row([round(T1,4),round(Tmax,4),round(T2,4)])
    print("Temperatures points")
    print(Table2,'\n')

    Table3=PrettyTable(["ω","δ","τ","Cω","Cδ","Cτ"])
    Table3.add_row([round(omega,4),round(thelta,4),round(tau,4),round(Comega,4),round(Cthelta,4),round(Ctau,4)])
    print("Ca and a parameters with a=ω,δ,τ\n")
    print(Table3,'\n')

    Table4=PrettyTable(["μg","μg'","FR","zm","r"])
    Table4.add_row([round(mg,4),round(mg_tone,4),round(FR,4),round(zm,4),round(r_root,5)])
    print("Significant parameters\n")
    print(Table4,"\n")

    Table5=PrettyTable(["Eτ","Eδ","Eω","Error %"])
    Table5.add_row([round(Etau,4),round(Ethelta,4),round(Eomega,4),round(Error,4)])
    print("Activation Energy\n")
    print(Table5,'\n')   
    print(s1)
    
    #File creation
    #output1="Initial conditions\n"+str(Table1)+"\n"
    #output2="Temperatures points in Kelvin\n"+str(Table2)+"\n"
    #output3="Ca and a parameters with a=ω,δ,τ\n"+str(Table3)+"\n"
    #output4="Significant parameters\n"+str(Table4)+"\n"
    #output5="Activation Energy\n"+str(Table5)

    #outputs=[output1,output2,output3,output4,output5]
    #file_path="C:\\Users\\User\\Python_files\\Thermoluminescence Glow Curve\\LOTOR.txt"

    #with open(file_path, "w") as file:
    #  for i in outputs:
    #       file.write(i)
    
L_OTOR()
