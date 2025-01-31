# Using the magic encoding
# -*- coding: utf-8 -*-



import time
import os
import numpy as np
import scipy.optimize
import csv
import datetime
import analyze.movingaverage
import analyze.constants

import analyze.current


class CurrentReadings(object):
    def __init__(self, files=(), roll_missaligned=False):
        self.readings = dict()
        self.fill_number = None
        self.fit = None
        self.chi2r = None
        self.singlebunch = False
        self.offset_subtracted = False
        self.offsetcurrent = 0.
        self.missaligne = dict()
        self.fit_fun = None
        self.rel_sigma_start = 0.01* np.ones(184)


        # self.rel_sigma_start = 0.01

        for filepath in files:
            #try:
            self.read_current(filepath, roll_missaligned)
            #except Exception:
            #    continue

        if self.offsetcurrent != 0. and not self.offset_subtracted:
            self.subtract_offset_current()

    def set_fillnumber(self, fillnumber):
        try:
            self.fill_number = fillnumber
            # sigma_start = {6212: 0.013,
            #                6258: 0.06,
            #                6288: 0.0067,#mean:0.008,#min:0.0047
            #                6284: 0.0117,
            #                6283: 0.0169,
            #                6292: 0.0086,
            #                6296: 0.006}
            sigma_start = {6212: 0.013 * np.ones(184),
                           6258: 0.06* np.ones(184),
                           6288: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.021124141684149264, 0.011149027220132335, 0.0084651582158146721, 0.007849235531498102, 0.0068428976445520017, 0.0057379420120428192, 0.0061034639663567853, 0.0065475656911149031, 0.0056192191288974005, 0.0057518779352177616, 0.0091804956407976011, 0.0065218932412641198, 0.0083431887520382143, 0.0063096820006714998, 0.0085607779018665193, 0.0069817491263137865, 0.0080270162383491942, 0.0064812203264589122, 0.0084422009931150871, 0.0067668044439671065, 0.0089421260572168683, 0.0070997018542808127, 0.0092576720030104736, 0.0068599434057003538, 0.0094247399926814975, 0.008569574945484347, 0.0088260875154516846, 0.0066616353310924364, 0.009397234612803269, 0.0067835986346999723, 0.010083535181118581, 0.0067549066887012416, 0.0096216136375832303, 0.0063883745370002269, 0.0080485667080720125, 0.0060086346033934581, 0.0097543569030365766, 0.0093222113192694706, 0.010467904883688794, 0.018196863131170974, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.04588314677411235, 0.016256402219058878, 0.0092517270538756448, 0.0069831108210220037, 0.0053223121825428195, 0.0049668329590179146, 0.0047660674322574778, 0.0048163276700018496, 0.0048141505210826455, 0.0049163352861094712, 0.0049073290481191642, 0.0050158879081156097, 0.0050793497223218633, 0.0051638400943766882, 0.0051201323878545526, 0.0052141357036312053, 0.0052131436821524864, 0.0052612635978930931, 0.0052421443078396312, 0.0052937476471778379, 0.0053127643477323665, 0.0053395831165750007, 0.005331229689719579, 0.005413319619607667, 0.0053975258015000445, 0.0055338216224037869, 0.0055105829041948128, 0.005620017735679955, 0.0054469430461216699, 0.005544443578553015, 0.0054955010262354529, 0.0054350234143351172, 0.0053330489116424221, 0.0052159795327033813, 0.0049046132171599041, 0.0049277827824456381, 0.0049153849252865566, 0.0052266553380777205, 0.0060863697733514587, 0.009811468853477787, 0.015244540097646505, 0.041702882811414953, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.021002657004177173, 0.011185934253567072, 0.0082467390594255282, 0.0074343791886277008, 0.0064614369145467952, 0.0072959666641624239, 0.0063094308140633588, 0.0060235499277189004, 0.0061980428431151428, 0.0066447016256250893, 0.006772388640772539, 0.0067097473519039542, 0.0065234195281711812, 0.0067073320360042776, 0.006639573466254399, 0.0076306980859034935, 0.006618164123094258, 0.0080329707109063429, 0.0065468640584666841, 0.0070108577099697381, 0.0062079477843265006, 0.0070986284999993328, 0.0065646142873149495, 0.0080366016667613015, 0.0066592715821202682, 0.0081156106610997241, 0.0092221731512832916, 0.008130081300813009, 0.0077398546205800665, 0.008473967705872297, 0.0076857162651786069, 0.007725060069817837, 0.0068376168275576339, 0.0071987170150086786, 0.0064500022816887104, 0.0074212652302318247, 0.0065663123251827995, 0.0092919790017865714, 0.010344520084234973, 0.018537599944001618, 0, 0],
                           6284: [0, 0, 0, 0, 0.041380294430118397, 0.015491313744672436, 0.021693045781865615, 0.011822115035531811, 0.012397752739056736, 0.011149027220132335, 0.020961090407515925, 0.011225331376673432, 0.012044696487560029, 0.011114542054873942, 0.022343928108437591, 0.011294569126194955, 0.010677644182170513, 0.011020998474952911, 0.024310831916315757, 0.011385711699531493, 0.013903647892017406, 0.011145564251507057, 0.01742625780188712, 0.011490458263551401, 0.011781840740656826, 0.01102903908775031, 0.023782574707724703, 0.011390881148923625, 0.013755535567588148, 0.011116602147330214, 0.016599939908326297, 0.011445214125649873, 0.011930150634611943, 0.010823626487770476, 0.016319055872014415, 0.011180339887498949, 0.014786041468221177, 0.010546786488216739, 0.01499400359760168, 0.011230993587059569, 0.012266476510495826, 0.011328581360736095, 0.024098134635593994, 0.027017161347914959, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.04184869753186872, 0.015721249524260256, 0.02092434876593436, 0.011460988720255175, 0.012760874905367245, 0.010835692610757274, 0.014828245646514833, 0.011108368642573443, 0.018395909204638759, 0.010969159685411956, 0.022564684120326209, 0.010925862791407363, 0.018804435361115038, 0.01066003581778052, 0.016736548175114462, 0.011126231081268294, 0.013666610481827953, 0.010823626487770476, 0.021276595744680851, 0.011092638760662028, 0.011344607913018797, 0.010960590800535361, 0.015084140199287575, 0.011270154618002348, 0.015031222212815091, 0.011165694518168505, 0.014028957596690997, 0.011282342060163104, 0.013777712461369822, 0.011192237964551243, 0.021647252975540952, 0.011368775789164532, 0.01831858263618279, 0.010952041965851957, 0.021915916515945887, 0.011353378388169527, 0.01466628860721083, 0.011665030436874564, 0.022990024493585143, 0.026099787548594027, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.03197647396955397, 0.01329556846135674, 0.022803964585835359, 0.011154574690170912, 0.011363636363636364, 0.00931331260378521, 0.012143758664527462, 0.010121167175522697, 0.016692188131234503, 0.0097381555635059659, 0.011476072944630294, 0.010789552140380451, 0.015330284613208399, 0.010644322847548647, 0.011558569738602507, 0.010958616215778212, 0.018151839895669614, 0.01044731901345371, 0.012158110873601292, 0.010853548048545139, 0.01789140667411283, 0.010595812726429147, 0.010812864581305084, 0.010830607221477648, 0.019090088708030313, 0.011081735562292474, 0.016133229794181921, 0.011174056247636599, 0.018867924528301886, 0.010938928884083116, 0.015092727811824554, 0.011221090334505813, 0.016280082351376849, 0.010665491157310273, 0.011598150010653683, 0.010887588907733555, 0.019010582334423435, 0.01178429469439066, 0.020869596778242055, 0.022977882942966148, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           6283: [0, 0, 0, 0, 0, 0.018841113964807383, 0.011160130449295197, 0.022809896167307987, 0.037959480900056165, 0.03197647396955397, 0.032342311367657542, 0.019849711139180454, 0.018436507984120484, 0.015605961367365603, 0.013714079724757678, 0.022377469175267228, 0.027136267706462856, 0.02891574659831201, 0.029000739528287078, 0.023174488732966077, 0.016888615337236124, 0.018547162802488575, 0.041204282171516463, 0, 0.043396303660274617, 0.042796049251091289, 0.019555285163358464, 0.019976043113781049, 0.026046613053531162, 0.026324906324632819, 0.020327890704543543, 0.022021944790965527, 0.022709684225171733, 0.020765845843612775, 0.016907916618206113, 0.016985788839624958, 0.018430244519362142, 0.019062321291575583, 0.014535559317825367, 0.013509813377402483, 0.01353577835636781, 0.01526406277181659, 0.030964056111131625, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.037139067635410375, 0.018499495895605175, 0.010657613907987187, 0.011479853315963399, 0.0088704225876841906, 0.011709730320404808, 0.0084837205132501335, 0.011847809009550361, 0.0086435234033815107, 0.011530873445432159, 0.009362573578414227, 0.013171584244345806, 0.0088433266574594292, 0.011886246180448265, 0.010049870596186849, 0.017052337204298631, 0.035007002100700242, 0.019433051858411793, 0.014066381153285495, 0.023168268228563865, 0.02891574659831201, 0.027735009811261455, 0.033864273073929821, 0.0349002406379888, 0.029086486358157505, 0.023695618019100723, 0.028104971365967134, 0.023044900641260895, 0.019683666479473904, 0.023550608702786628, 0.024536645240595824, 0.023511520528619944, 0.03137279025690793, 0.03137279025690793, 0.040790850822400207, 0.020589036786063403, 0.017650112740455196, 0.026379807127383248, 0.015276525413351826, 0.025523828044507795, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.030700277992275839, 0.01531589335482564, 0.0091991220776813602, 0.010962566453052481, 0.0068903278416741808, 0.010692282796400105, 0.0056746780314490576, 0.0098937186910675618, 0.0071812475327089134, 0.010507123617866617, 0.0058983963555528356, 0.011093321279583498, 0.0069241053725020638, 0.010576830178027529, 0.0089958763358334078, 0.011643660820815558, 0.010930430596399216, 0.014766683779622342, 0.014869169292381494, 0.012551092808470124, 0.009619832681781286, 0.011616917255955381, 0.0086104573789786334, 0.011936948441792959, 0.0099820484546577857, 0.018718584988357507, 0.0077729859135006593, 0.011460988720255175, 0.013124987182635961, 0.011638927945819883, 0.0095433055718978036, 0.013689641954834839, 0.016211480171659538, 0.011665030436874564, 0.012139284012153828, 0.018663083698528475, 0.014664711502135329, 0.026361468712148239, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           6292: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.034441829515914534, 0.014907119849998599, 0.0092351419307922025, 0.0070966619475457448, 0.0062770298096225781, 0.0058707852800313218, 0.0057488356194028249, 0.0056821850702815369, 0.0056954405744618127, 0.0057820858230844653, 0.0058195625168646486, 0.0058472533715623496, 0.0058241997371329046, 0.0060666255350849746, 0.0069572056568561269, 0.0079450200133319958, 0.013112569439434401, 0.0062863249869485188, 0.0085935900256004678, 0.0071403074882541006, 0.0097851292922567727, 0.010891462805350444, 0.011849472443606932, 0.007854318214880562, 0.012335342787286209, 0.0082678517368544504, 0.012387285278916001, 0.011455723277057845, 0.024536645240595824, 0.010583936671670079, 0.010709437512811186, 0.0087895319241828741, 0.010653379535009885, 0.010132071090994062, 0.020974918506045256, 0.0068854261352547883, 0.014950372381037679, 0.013240678533309402, 0.01249024580421098, 0.021417646843905968, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.032668576019240059, 0.014132246511151978, 0.0087709175425403126, 0.0068641439319541333, 0.0060613853521608984, 0.0057858590520804233, 0.0056371362099047837, 0.0055997150937438607, 0.0055912183955001988, 0.0055926172492612655, 0.0056376736857479768, 0.0056674737094120076, 0.0056462042599669613, 0.0056929481007740859, 0.0057093475165272758, 0.0057332249381851878, 0.0056927636032418887, 0.0059319243522645733, 0.0061830966308910138, 0.0063098076052261641, 0.0059385103286194868, 0.0060430959072100467, 0.0075899278660478357, 0.0099024463054301799, 0.0096488963620454012, 0.011257293218825929, 0.013379797308309913, 0.0071574791789233531, 0.0079089077971181612, 0.0093662689260948229, 0.037529331252040075, 0.00704382566719033, 0.018124985839860344, 0.013596950274133403, 0.0068711079218319706, 0.010812864581305084, 0.010674602031568248, 0.018831089428867735, 0.016900670885446459, 0.02084690996125416, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.034794450031961049, 0.014962081765396863, 0.0091851416409132157, 0.0070611890574792495, 0.0062372219113766081, 0.0058686618291763594, 0.0056902746409049805, 0.0057142857142857143, 0.0057237318259103524, 0.0057048862167512369, 0.0057388868247498278, 0.0057459878282638908, 0.0058853065177616737, 0.0068568786624995937, 0.012901346454777697, 0.0059762497663706591, 0.0060569363018314355, 0.011239502991174948, 0.036369648372665396, 0.0067177667621255693, 0.013771178713925343, 0.021417646843905968, 0.013563140979519323, 0.0087177630160990248, 0.010226467213413207, 0.022107884414269093, 0.012993589245171131, 0.0093266702530829578, 0.013435230372511476, 0.011437725271791937, 0.027017161347914959, 0.0083382566744522081, 0.012108099334382623, 0.014859316721917632, 0.007660193134288098, 0.0075790205976497419, 0.011169176290219258, 0.01823312393723682, 0.018556740475630138, 0.021677749238102999, 0],
                           6296: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.022000616025873207, 0.011251591157819946, 0.0077758052773632298, 0.0065345521264453914, 0.0060018128212493728, 0.0058088506014321289, 0.0057600781532705795, 0.0057455136077386613, 0.0057465570479223547, 0.0057818925230986836, 0.0057719637067451274, 0.0058593861759028698, 0.0058395716247378378, 0.0059167454544093549, 0.0058815389436715458, 0.0059116772229117886, 0.0059162276906407499, 0.0059161241541971646, 0.0058937845095906908, 0.005887652175696657, 0.0059343262159823546, 0.0059506944865722394, 0.006002461513934557, 0.0060159150993849813, 0.0060540496671913115, 0.0061215057131932355, 0.0061008509145133276, 0.0061379733216883857, 0.0061423717266395625, 0.006101532256637128, 0.0059880239520958087, 0.0059026076988189773, 0.0058241997371329046, 0.0057583589264964341, 0.0057778376964053198, 0.006058603547397387, 0.0068266145103174289, 0.0088041678230544439, 0.014574100933227224, 0.033557802760701215, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.03494282789073061, 0.015760250451281974, 0.011275884962655315, 0.008221893962045838, 0.0072191346970701667, 0.0073964577969760385, 0.0075431428640471429, 0.0073037465412545844, 0.0073097930937524131, 0.0070351048198854542, 0.0073039413571824204, 0.0075846865041755392, 0.0078837977454030376, 0.0078163174935090302, 0.0079686966939552916, 0.0074601941585556728, 0.0082080341801640932, 0.0077711080404495184, 0.008159256327467744, 0.0083912226725987339, 0.0079674319619819963, 0.0083600819425807549, 0.0080932535644611114, 0.0084703190651408845, 0.0084754893650463119, 0.0091173197560739903, 0.0086200490689069802, 0.0083203429383219649, 0.0080187537387448014, 0.007705541583566294, 0.0077580009276242258, 0.0082230057800309134, 0.0081190872710360778, 0.0078473018650201554, 0.0076676204927865624, 0.0076240420516482657, 0.0077603366196619802, 0.009740003060309442, 0.016406675593723583, 0.037450294313656908, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.019687480773953943, 0.010268434267346985, 0.0072833772168248477, 0.0066645935599922261, 0.0061601772579547732, 0.0058878562785229082, 0.0055559842745866025, 0.0054578030662501839, 0.0054949202336380745, 0.0054651336809225376, 0.0054592668292980703, 0.0054475087565709424, 0.0054987402454453639, 0.0054932618403385081, 0.0054999876250417653, 0.0055345843645850396, 0.0055331438940252114, 0.0055182125652006309, 0.0055178765293591838, 0.0055495638907834257, 0.0055471726471135489, 0.0055635460123827684, 0.0055694103668264532, 0.0056111634042803988, 0.0056183321871936844, 0.0056554970930292906, 0.0056694772192139101, 0.0056910117708570207, 0.0056795267379394221, 0.00564009423096951, 0.0055501621850464096, 0.0054556908189624526, 0.0053248015533360008, 0.0053227645357286392, 0.0052998405301976333, 0.0054916049476663419, 0.006152477470860793, 0.007873527603475804, 0.012480514407482947, 0.028386647790182979, 0]}
            self.rel_sigma_start = sigma_start[fillnumber]
        except KeyError:
            print('No relativ error on current measurement present for this fill! Using 0.01.')

    def read_current(self, filepath, roll_missaligned, timestamp=None):
        import matplotlib.pyplot as plt
        data = current.readCurrent(filepath)

        if not timestamp:
            fileName = os.path.basename(filepath)
            filestring = fileName.split('.')[0]
            if filestring[0] != 'f':
                timestring = filestring
                timestamp = time.mktime(time.strptime(timestring, "%Y_%m_%d_%H_%M_%S"))
            else:
                timestring = filestring.split('_')[1]
                timestring = timestring.split('s')[0]
                timestamp = time.mktime(time.strptime(timestring, "%Y-%m-%dT%Hh%Mm%S"))

            # assert abs(current.readinfo(filepath, info='MeasTime') - timestamp) < 1.0, \
            #     'Measurement time in file differs by more than 1 second from the time in filename!\n%f,%f' %(current.readinfo(filepath,info='MeasTime'), timestamp)

        if roll_missaligned:

            info = current.readinfo(filepath, info='MissalignedBucket')
            if info is None:
                info = current.readinfo(filepath, info='MissalignBucket')
                if info is None:
                    info=0

            if len(list(self.readings.keys())) != 0:
                ref = self.readings[min(self.readings.keys())]
                h = ref.shape[0]
                chi2 = np.array([np.sum((np.roll(data, shift) - self.readings[min(self.readings.keys())]) ** 2) for shift in range(h)])
                shift = chi2.argmin()
                print((info,shift))

                if shift != -info and shift != constants.ANKA.h-info:
                    print('fit suggests other shift (maybe due to empty fillingpattern at begin of fill)\n but rotating by missaligned bucket')
                    #info = -shift
                    #print('rotating by fit not by missaligned info!')

            # plt.figure()
            # plt.plot(data)
            data = np.roll(data, -info)
            self.missaligne[timestamp] = info
            # plt.plot(data)

        self.readings[timestamp] = data
        self.fit = None

    # linear interpolation
    def get_filling_pattern(self, timestamp):
        assert min(self.readings.keys()) <= timestamp <= max(
            self.readings.keys()), 'Not within valid time range from %i to %i (timestamp: %i)' % (
            min(self.readings.keys()), max(self.readings.keys()), timestamp)

        times = np.array(list(self.readings.keys()))
        times.sort()

        try:
            return self.readings[timestamp]
        except KeyError:
            pass

        right_timestamp_idx = np.where((times - timestamp) > 0)[0][0]
        right_timestamp = times[right_timestamp_idx]
        left_timestamp = times[right_timestamp_idx - 1]

        left_pattern = self.readings[left_timestamp]
        right_pattern = self.readings[right_timestamp]

        return left_pattern + (right_pattern - left_pattern) * (timestamp - left_timestamp) / (
            right_timestamp - left_timestamp)

    def get_bucket_smooth(self, bucket, timestamp_list):

        time = sorted(self.readings.keys())
        def lowpass_filter(freq, ny, filter_order=4):
            b, a = scipy.signal.butter(filter_order, freq / (1 / (time[10]-time[9]) / 2), btype='low')
            return np.asarray(scipy.signal.filtfilt(b, a, ny))
        bucket_current = [self.readings[x][bucket] for x in time]
        # smooth_bucket_current = movingaverage.movingaverage(bucket_current, 50)
        smooth_bucket_current = lowpass_filter(0.0005, bucket_current)
        bucket_current_raw = bucket_current
        smooth_bucket_current_dict = {t: smooth_bucket_current[i] for i, t in enumerate(time)}

        bucket_current = np.zeros(len(timestamp_list))
        times = np.array(list(smooth_bucket_current_dict.keys()))
        times.sort()
        temp_current = np.array([smooth_bucket_current_dict[t] for t in times])
        bucket_current = np.interp(timestamp_list, times, temp_current)

        # for i, timestamp in enumerate(timestamp_list):
        #     times = np.array(smooth_bucket_current_dict.keys())
        #     times.sort()
        #     right_timestamp_idx = np.where((times - timestamp) > 0)[0][0]
        #     right_timestamp = times[right_timestamp_idx]
        #     left_timestamp = times[right_timestamp_idx - 1]
        #
        #     left_current = smooth_bucket_current_dict[left_timestamp]
        #     right_current = smooth_bucket_current_dict[right_timestamp]
        #
        #     bucket_current[i] = left_current + (right_current - left_current) * (timestamp - left_timestamp) / (right_timestamp - left_timestamp)

        FitModel =  temp_current
        DoF = len(times)#?
        rel_sigma = np.sqrt(np.abs(bucket_current_raw[0])/bucket_current_raw)*self.rel_sigma_start[bucket]
        print((len(temp_current), len(bucket_current_raw)))

        Chi_square = np.sum((bucket_current_raw-FitModel)**2/(rel_sigma*bucket_current_raw)**2)
        # Chi_square = np.sum((Y-FitModel)**2/(FitModel))
        Chi_square_red = Chi_square / DoF

        if self.chi2r is None:
            self.chi2r={}
        self.chi2r[bucket] = Chi_square_red
        print(('smooth bucket '+ str(bucket)+' chi2_red '+ str(Chi_square_red)+' chi2 ' + str(Chi_square)))

        return bucket_current

    def get_bucket_fit(self, bucket, timestamp):
        # assert min(self.readings.keys()) <= timestamp <= max(self.readings.keys()), 'Not within valid time range'
        if self.fit == None:
            print('calculate fit')
            self.calculate_fit()
            print('calculating done')

        return self.fit[bucket](timestamp)

    # fit instead of linear interpolation
    def get_filling_pattern_fit(self, timestamp):
        return np.array([self.get_bucket_fit(bucket, timestamp) for bucket in range(constants.ANKA.h)])

    def calculate_fit(self):
        import matplotlib.pyplot as plt
        import scipy.interpolate

        def exponential(x, a, b, c):
            return a * np.exp(b * x) + c

        def double_exponential(x, a, b, c, d, f):
            return a * np.exp(x * b) + c * np.exp(x * d) + f

        def more_exponential(x, a, b, c, d, f, g):
            return a * np.exp(x * b) + c * np.exp(x * d) + f - abs(g) * x

        def polynomial(x, a, b, c, d, e, f, g, h, i, j, k):
            return a * x ** 5 + b * x ** 4 + c * x ** 3 + d * x ** 2 + e * x + f + g * x ** 6 + h * x ** 7 + i * x ** 8 + j * x ** 9 + k * x ** 10

        # fit_fun_dict = {'exponential': (exponential, [0.3, -0.4e-4, 0.001]),
        fit_fun_dict = {'exponential': (exponential, [0.3, -0.001, 0.1]),
                        'part_exponential': (exponential, [0.3, -0.001, 0.1]),
                        # 'double_exponential': (double_exponential, [0.3, -1e-4, 0.05, -1e-4, 0]),
                        'double_exponential': (double_exponential, [0.2, -1.4e-4 , 0.25, -6.14e-04, 7.1e-2]),
                        'part_double_exponential': (double_exponential, [0.3, -0.001 , 0.01, -1e-04, 0.1]),
                        'more_exponential': (more_exponential, [0.3, -1e-4, 0.05, -1e-4, 0, 0]),
                        'polynomial': (polynomial, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])}
                        # 'polynomial': (polynomial, [-2.06e-20,2.95e-16,-2.75e-12,1.65e-08,-6.48e-05,1.76e-01,9.39e-25,-2.77e-29,5.09e-34,-5.29e-39,2.37e-44]),}

        current_range_dict = {6212: [0.14, 0.08],
                              6258: [0.26, 0.16],
                              6288: [0.16, 0.08],
                              6284: [0.09, 0.05],
                              6283: [0.20, 0.15],
                              6292: [0.17, 0.08],
                              6296: [0.27, 0.21]}
        if self.fit_fun=='part_exponential' or self.fit_fun=='part_double_exponential':
            current_range = current_range_dict[self.fill_number]

        if self.fit_fun is None:
            self.fit_fun = 'double_exponential' if self.singlebunch else 'polynomial'

        print(('using %s interpolation' %(self.fit_fun)))

        # assert self.fit_fun in fit_fun_dict, 'Scaling function must be one of %s' % fit_fun_dict.keys()

        X = np.array(sorted(self.readings.keys()))
        print((len(X)))

        self.fit = dict()
        self.chi2r = dict()

        # plt.figure()
        # plt.title('chi2_red ' + str(self.fill_number))
        # ax1 =plt.gca()
        # ax1.set_ylabel('chi2_red')
        # # ax2=plt.twinx(ax1)
        # ax2.set_ylabel('initial current')
        print((self.rel_sigma_start))
        # TODO: do the same for singlebunch? does it work with all zeros for one bunch?
        for bucket in range(constants.ANKA.h):
            X = np.array(sorted(self.readings.keys()))
            X_old = X[:]
            Y_with_zero = np.asarray([self.readings[x][bucket] for x in X])
            # current less than 0 is unphysical:
            # Y = np.where(Y_with_zero < 0, 0, Y_with_zero)
            Y =Y_with_zero
            rel_sigma = np.sqrt(np.abs(Y_with_zero[0])/Y)*self.rel_sigma_start[bucket]

            if self.fit_fun=='part_exponential' or self.fit_fun=='part_double_exponential':
                ind = np.logical_and(Y>current_range[1],Y<current_range[0])

                if np.sum(ind) > 10:
                    Y=np.array(Y[ind])
                    X=np.array(X[ind])
                    rel_sigma = np.array(rel_sigma[ind])
                    print('partial')

            X2 = X - X.min()

            # curvefit, pcov = scipy.optimize.curve_fit(fit_fun_dict[self.fit_fun][0], X2, Y, p0=fit_fun_dict[self.fit_fun][1], maxfev=30000, sigma=rel_sigma, absolute_sigma='False', ftol=1e-6)

            if False:#self.fit_fun == 'polynomial':
                fit_result = np.polyfit(X2, Y, deg=10)
                self.fit[bucket] = lambda x, fit_result=fit_result, X=X: np.polyval(fit_result, x - X.min())
                FitModel = self.fit[bucket](X)
                DoF = len(X2) - len(fit_result)
            else:
                if True:
                    #curvefit, pcov = scipy.optimize.curve_fit(fit_fun_dict[self.fit_fun][0], X2, Y, p0=fit_fun_dict[self.fit_fun][1], maxfev=30000, sigma=rel_sigma, absolute_sigma='False', ftol=1e-6)

                    if self.fit_fun == 'test':
                        curvefit, pcov = scipy.optimize.curve_fit(fit_fun_dict['exponential'][0], X2, Y, p0=fit_fun_dict['exponential'][1], maxfev=30000, sigma=rel_sigma, absolute_sigma='False', ftol=1e-6)
                        a, b, c = curvefit
                        curvefit, pcov = scipy.optimize.curve_fit(fit_fun_dict['double_exponential'][0], X2, Y, p0=[a, b, 0, b, c], maxfev=30000, sigma=rel_sigma, absolute_sigma='False', ftol=1e-6)
                        self.fit_fun = 'double_exponential'
                    elif False and self.fit_fun=='exponential':
                        import probfit
                        import iminuit
                        # Define a chi^2 cost function
                        mchi2 = probfit.Chi2Regression(fit_fun_dict[self.fit_fun][0], X2, Y, rel_sigma*Y)
                        a,b,c= fit_fun_dict[self.fit_fun][1]
                        minuit = iminuit.Minuit(mchi2, a=a, b=b, c=c, print_level=0, pedantic=False)
                        minuit.migrad()  # MIGRAD is a very stable robust minimization method
                        # minuit.hesse()
                        # mchi2.draw(minuit)
                        chi2_red  = mchi2(*minuit.args)/mchi2.ndof
                        # print(minuit.values)
                        print(('bucket:', bucket, 'Chi_square_red:',chi2_red))
                        curvefit = np.array([minuit.values['a'],minuit.values['b'],minuit.values['c']])#,minuit.values['d'],minuit.values['f']])
                    else:
                        print('standard fit')
                        curvefit, pcov = scipy.optimize.curve_fit(fit_fun_dict[self.fit_fun][0], X2, Y, p0=fit_fun_dict[self.fit_fun][1], maxfev=30000, sigma=rel_sigma, absolute_sigma='False', ftol=1e-6)


                else:
                    def min_fun(x0, x, y, fun):
                        return sum((y-fun(x, *x0))**2)

                    if self.fit_fun == 'test':
                        result = scipy.optimize.minimize(min_fun, x0=fit_fun_dict['exponential'][1], args=(X2, Y, fit_fun_dict['exponential'][0]), method='Nelder-Mead')
                        a, b, c = result.x
                        result = scipy.optimize.minimize(min_fun, x0=[a, b, 0, b/2, c], args=(X2, Y, fit_fun_dict['double_exponential'][0])) #, method='Nelder-Mead')
                        curvefit, pcov = result.x, None

                        # curvefit, pcov = scipy.optimize.curve_fit(fit_fun_dict['double_exponential'][0], X2, Y, p0=result.x, maxfev=30000, sigma=rel_sigma, absolute_sigma='False', ftol=1e-8)

                        self.fit_fun = 'double_exponential'
                    else:
                        result = scipy.optimize.minimize(min_fun, x0=fit_fun_dict[self.fit_fun][1], args=(X2, Y, fit_fun_dict[self.fit_fun][0]), method='Nelder-Mead')
                        curvefit, pcov = result.x, None
                    # print(result)
                    # assert result.success


                temp = fit_fun_dict[self.fit_fun][0]

                # print(self.fit_fun, bucket)
                if bucket == 47 or bucket==48:
                    print(curvefit)
                # if bucket == 173:
                #     print(X2,Y, rel_sigma*Y)
                # self.fit[bucket] = lambda x, curvefit=curvefit: fit_fun_dict[self.fit_fun][0](x - X.min(), *curvefit)
                self.fit[bucket] = lambda x, curvefit=curvefit, x_offset=X.min(), fun=temp: fun(x - x_offset, *curvefit)
                # self.fit[bucket] = lambda x, curvefit=curvefit, X=X, fun=temp: fun(x - X.min()- X3.min(), *curvefit)
                #
                FitModel = temp(X2, *curvefit)
                DoF = len(X2) - curvefit.size

            Chi_square = np.sum((Y-FitModel)**2/(rel_sigma*Y)**2)
            # Chi_square = np.sum((Y-FitModel)**2/(FitModel))
            Chi_square_red = Chi_square / DoF
            # popt_err = np.sqrt(np.diag(pcov)/Chi_square_red)
            # if not np.isnan(Chi_square):# and Y[0] >= 0.02:
            #     print('bucket:', bucket, 'Chi_square_red:', Chi_square_red)#, Chi_square_red-chi2_red)#, 'Chi_square:', Chi_square)#, 'popt_err:',popt_err)
            #     plt.scatter(Y_with_zero[0], Chi_square_red)
            #     plt.xlabel('initial bunch current / mA')
            # ax1.scatter(bucket, Chi_square_red)
                # ax2.scatter(bucket, Y_with_zero[0], color='r')
                # plt.xlabel('bucket number')


            self.chi2r[bucket] = Chi_square_red
        # plt.savefig('chi.png')
            # W = np.ones_like(X)*1./0.0002
            # curvefit = scipy.interpolate.UnivariateSpline(X2, Y, s=0.001)
            # print(bucket, curvefit)

            # self.fit[bucket] = lambda x, curvefit=curvefit: curvefit(x - X.min())

            # self.fit[bucket] = scipy.interpolate.UnivariateSpline(X, Y, w=W)#s=0.001)
            # self.fit[bucket] = scipy.interpolate.UnivariateSpline(X, Y, s=0.001)

            # Yfit = np.array([self.get_bucket_fit(bucket, x) for x in X])
            # fehler = (np.sum((Yfit-Y)**2)/len(X))**0.5
            # offset = np.sum((Yfit-Y)/len(X))

            # print(bucket,fehler,offset)

            # plt.figure()
            # plt.plot(X2, Y, '.')
            # plt.plot(X2, exponential(X2, *curvefit), '-')
            # plt.grid()
            # plt.show()

    def rotate_to_reference(self, ref, timestamp=None):
        h = ref.shape[0]
        timestamp = timestamp if timestamp else min(self.readings.keys())
        filling_pattern = self.get_filling_pattern(timestamp)

        chi2 = np.array([np.sum((np.roll(filling_pattern, shift) - ref) ** 2) for shift in range(h)])
        shift = chi2.argmin()

        # Roll all current readings
        self.readings = {ts: np.roll(data, shift) for ts, data in list(self.readings.items())}
        self.fit = None

    def subtract_offset_current(self):
        if not self.offset_subtracted and self.offsetcurrent != 0:
            if self.singlebunch:
                # find bucket with singelbunch
                single_row = list(self.readings.values())[0]
                bucket = np.where(single_row != 0)
                for ts, data in list(self.readings.items()):
                    data[bucket] = data[bucket] - self.offsetcurrent
                    self.readings[ts] = data

                self.offset_subtracted = True
            else:
                self.readings = {ts: data / data.sum() * (data.sum() - self.offsetcurrent) for ts, data in
                                 list(self.readings.items())}
                self.offset_subtracted = True

    def timeoffset_PH(self, timeoffset):
        for key in list(self.readings.keys()):
            self.readings[key + timeoffset] = self.readings.pop(key)

    def set_offsetcurrent(self, value):
        assert self.offsetcurrent == 0., 'The offsetcurrent of %f will be overwritten!' % self.offsetcurrent
        assert not self.offset_subtracted, 'An offsetcurrent was already subtracted before!'
        self.offsetcurrent = value
        self.subtract_offset_current()

    def plot(self, lines=(), german=False):
        import matplotlib
        import matplotlib.pyplot as plt
        font = {'size': 19}
        matplotlib.rc('font', **font)

        plt.figure()
        X = sorted(self.readings.keys())
        Y = [self.readings[x].sum() for x in X]
        plt.plot(X, Y, 'o')
        plt.title('beam current over time')
        plt.xlabel('Unix Timestamp')  # , size=12)
        plt.ylabel('Current / mA')  # , size=12)
        if german:
            plt.xlabel('Unixzeit / s')  # , size=12)
            plt.ylabel('Strom / mA')  # , size=12)
            # plt.title(u'Strahlstrom über Zeit')
        plt.grid()
        for line in lines:
            plt.axvline(line, color='red')

        # plt.savefig('beam current over time'+'.pdf')
        # plt.figure()
        # tmp = [self.missaligne[x] for x in X]
        # plt.plot(tmp)

        for line in lines:
            plt.figure(figsize=(12, 6))
            plt.bar(np.arange(constants.ANKA.h) - 0.5, self.get_filling_pattern(line))
            # plt.plot(self.get_filling_pattern_fit(line))#, 'red')
            # plt.plot(self.get_filling_pattern(line))#, 'red')
            # plt.title('Filling pattern')# at %f' % line)
            plt.xlabel('bucket number')
            plt.ylabel('bunch current / mA')
            if german:
                plt.xlabel('Bucketnummer')  # , size=12)
                plt.ylabel('Bunchstrom / mA')  # , size=12)
                # plt.title(u'Füllmuster')# at %f' % line)
            plt.xlim(xmin=0, xmax=constants.ANKA.h)
            # plt.grid()
            # plt.yscale('log')

            # plt.savefig('filling-pattern-at-timestamp-'+str(line)+'.pdf')

    def plot_bucket(self, buckets, lines=(), linear=False, german=False, xaxis='unix', notitle='False'):
        import matplotlib.pyplot as plt
        import matplotlib
        font = {'size': 19}
        matplotlib.rc('font', **font)

        X = sorted(self.readings.keys())
        for bucket in buckets:
            # fig = plt.figure(figsize=(10, 5), tight_layout=True)
            fig = plt.figure(figsize=(12, 5), tight_layout=True)
            ax=plt.subplot(111)
            X_long = np.linspace(X[0], X[-1], 10 * len(X))
            Y = [self.readings[x][bucket] for x in X]
            error=Y *  np.sqrt(np.abs(Y[0])/Y)*self.rel_sigma_start[bucket]
            if linear:
                Yfit = Y
                X_long = np.asarray(X)
            else:
                Yfit = [self.get_bucket_fit(bucket, x) for x in X_long]
                # Yfit = np.array(self.get_bucket_smooth(bucket, X_long))

            X = np.array(X)
            if xaxis == 'time':
                X_new = [datetime.datetime.fromtimestamp(int(tmp)) for tmp in X]
                X_long_new = [datetime.datetime.fromtimestamp(int(tmp)) for tmp in X_long]
                plt.xlabel('Zeit') if german else plt.xlabel('Time')
                ax = plt.gca()
                ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M:%S'))
                for line in lines:
                    plt.axvline(datetime.datetime.fromtimestamp(int(time)), color='red')
            elif xaxis == 'min':
                X_new = (X - X[0]) / 60.
                X_long_new = (X_long - X_long[0]) / 60.
                plt.xlabel('Zeit seit Messbeginn / min') if german else plt.xlabel('Time since Measurement Start / min')
                for line in lines:
                    plt.axvline((line - X[0]) / 60., color='red')
            else:
                X_new = X
                X_long_new = X_long
                plt.xlabel('Unixzeit / s') if german else plt.xlabel('Unix Timestamp / s')
                for line in lines:
                    plt.axvline(line, color='red')

            # plt.plot(X_new, Y, '.')#,alpha=0.5)
            plt.errorbar(X_new, Y, yerr=error, fmt='.')#,alpha=0.5)
            plt.plot(X_long_new, Yfit, 'r-',zorder=10)

            # plt.grid()
            # plt.yscale('log')
            if not notitle:
                plt.title('Bucket No. %i' % bucket)
            plt.ylabel('Bunchstrom / mA') if german else plt.ylabel('Bunch Current / mA')
            # plt.text(0.5, 0.95,
            #                      '$\chi^{2}_{red} = %.3f$' % (self.chi2r[bucket]),
            #                      #(current_readings.get_bucket_fit(args.bucket, timestamp)),
                                 # horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

                # plt.savefig(str(bucket)+'.pdf')

    def parse_header(self, input=False):
        if input:
            assert 'singlebunch' in input[0], 'header not present or different format'
            header = [x for x in input if x != '0.0']
            header_dict = {c.split('=')[0]: c.split('=')[1] for c in header}
            self.singlebunch = header_dict['singlebunch'] == 'True'
            assert self.offsetcurrent == 0., 'offsetcurrent of %f will be overwritten!' % self.offsetcurrent
            self.offsetcurrent = float(header_dict['offsetcurrent'])
            self.offset_subtracted = header_dict['offset_subtracted'] == 'True'
        else:
            tmp = list()
            tmp.append('singlebunch=%s' % str(self.singlebunch))
            tmp.append('offsetcurrent=%s' % str(self.offsetcurrent))
            tmp.append('offset_subtracted=%s' % str(self.offset_subtracted))
            return tmp

    def save(self, filename):
        with open(filename, 'w') as f:
            w = csv.writer(f)
            # write header containing self.singlebunch, self.offsetcurrent, self.offset_subtracted
            header = self.parse_header()
            header.extend(np.zeros(185 - len(header)))
            w.writerow(header)
            # w.writerow(['time',]+['bucket_' + str(b) for b in range(184)])

            # write current values to file
            sortedkeys = sorted(self.readings.keys())
            for key in sortedkeys:
                row = list()
                row.append(key)
                row.extend(self.readings[key])
                w.writerow(row)

    def load(self, filename):
        print(filename)
        if filename.split('.')[-1] == 'csv':
            with open(filename, 'r') as f:
                reader = csv.reader(f)
                # get first line, header, containing information about type and offset current
                firstline = next(reader)
                try:
                    self.parse_header(firstline)
                except AssertionError as e:
                    print(('Information: %s' % e.message))
                    assert len(firstline) == constants.ANKA.h + 1, 'Not the correct number of columns'
                    row_float = [float(c) for c in firstline]
                    self.readings[row_float[0]] = np.array(row_float[1:])

                # read current values
                for row in reader:
                    assert len(row) == constants.ANKA.h + 1, 'Not the correct number of columns (%i) instead of 184+1' % len(row)
                    row_float = [float(c) for c in row]
                    self.readings[row_float[0]] = np.array(row_float[1:])
                # if offsetcurrent given subtract
                if self.offsetcurrent != 0. and self.offset_subtracted == False:
                    self.subtract_offset_current()

        if filename.split('.')[-1] == 'h5':
            print('loading processed h5 Fillingpattern File.')
            data, timestamps, attr = current.readCurrent(filename)
            print((data.size, timestamps.size))
            for row, t in enumerate(timestamps):
                self.readings[t] = data[row,:]
            self.fit = None


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Processing ANKA filling pattern files')
    parser.add_argument('infile', nargs='*', type=str)
    parser.add_argument('--save', type=str, default=None)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--timestamp', type=float, nargs='*', default=tuple())
    parser.add_argument('--bucket', type=int, nargs='*', default=tuple())
    parser.add_argument('--showplot', action='store_true')
    parser.add_argument('--mintime', type=float)
    parser.add_argument('--maxtime', type=float)
    parser.add_argument('--roll_missaligned', action='store_true')
    parser.add_argument('--german', '-g', action='store_true')
    parser.add_argument('--linearcurrent', action='store_true')
    parser.add_argument('--offsetcurrent', type=float)

    args = parser.parse_args()

    current_readings = CurrentReadings(args.infile, args.roll_missaligned)

    if args.load:
        current_readings.load(args.load)

    # manually insert offsetcurrent, must be done AFTER readin/load of current values!
    if args.offsetcurrent:
        if not args.save:
            print('The given offsetcurrent will not be saved, additionaly use --save to save to new csv file')
        current_readings.set_offsetcurrent(args.offsetcurrent)

    if args.mintime or args.maxtime:
        for key in list(current_readings.readings.keys()):
            if args.mintime and key < args.mintime:
                del current_readings.readings[key]
            if args.maxtime and key > args.maxtime:
                del current_readings.readings[key]

    if args.showplot:
        import matplotlib.pyplot as plt

        current_readings.plot(lines=args.timestamp, german=args.german)
        current_readings.plot_bucket(args.bucket, lines=args.timestamp, linear=args.linearcurrent, german=args.german)
        plt.show()

    for timestamp in args.timestamp:
        print((current_readings.get_filling_pattern(timestamp)))

    if args.save:
        current_readings.save(args.save)

        # print(current_readings.get_bucket_fit(1, 1382186268))
