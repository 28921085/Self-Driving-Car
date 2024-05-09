

class Fuzzy_System:
    def fuzzy_rule_1(self,front_distance):
        x=0 #LOW
        y=0 #MEDIUM
        z=0 #HIGH
        if front_distance<10:
            x=1
        elif front_distance>=10 and front_distance<=15:
            x=(15-front_distance)/5
        
        if front_distance>=10 and front_distance<20:
            y=(front_distance-10)/10
        elif front_distance>=20 and front_distance<=30:
            y=(30-front_distance)/10

        if front_distance>=30:
            z=1
        elif front_distance>=25 and front_distance<30:
            z=(front_distance-25)/5 

        return [x,y,z]
            
    def fuzzy_rule_2(self,right_distance,left_distance):
        dif = right_distance-left_distance
        x=0 #LOW
        y=0 #MEDIUM
        z=0 #HIGH
        if dif<-10:
            x=1
        elif dif>=-10 and dif<=0:
            x=(-dif)/10
        
        if dif>0 and dif<=15:
            y=(15-dif)/15
        elif dif<=0 and dif >=-15:
            y=(dif+15)/15

        if dif>=10:
            z=1
        elif dif>=0 and dif<10:
            z=(dif)/10

        return [x,y,z]

    def get_next_Th(self, inputs):
        p1=self.fuzzy_rule_1(inputs[0])
        p2=self.fuzzy_rule_2(inputs[1],inputs[2])
        #左轉或右轉
        angle = p2[0]*-1+p2[2]*1
        #轉彎的強度
        angle = angle*(30*p1[0]+16*p1[1]+2*p1[2])
        return angle