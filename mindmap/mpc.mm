<map version="1.0.1">
<!-- To view this file, download free mind mapping software FreeMind from http://freemind.sourceforge.net -->
<node CREATED="1588020622003" ID="ID_1170010713" MODIFIED="1588596045128" TEXT="MPC">
<node CREATED="1588020806676" ID="ID_577791587" MODIFIED="1588020934191" POSITION="right" TEXT="Optimal control problem">
<node CREATED="1588020845174" ID="ID_657880115" MODIFIED="1588020861615" TEXT="Finite-horizon optimal control problem (FHOCP)"/>
<node CREATED="1588020908484" ID="ID_854381527" MODIFIED="1588020928431" TEXT="Infinite-horizon optimal control problem (IHOCP)"/>
</node>
<node CREATED="1588111463320" ID="ID_1902138515" MODIFIED="1588111477186" POSITION="left" TEXT="elements">
<node CREATED="1588111489147" ID="ID_1940838981" MODIFIED="1588112304722" TEXT="stage cost $l$"/>
<node CREATED="1588111494407" ID="ID_1444443964" MODIFIED="1588112373238" TEXT="terminal cost $\mathcal F$ (often omitted) "/>
</node>
<node CREATED="1588432997491" ID="ID_1224387529" MODIFIED="1588596047086" POSITION="right" TEXT="How you can play with MPC">
<node CREATED="1588433850397" ID="ID_361438757" MODIFIED="1588433879906" TEXT="Play with the optimizer"/>
<node CREATED="1588596048733" ID="ID_1737968742" MODIFIED="1588600572639" TEXT="Approach the model">
<node CREATED="1588596078466" ID="ID_1165294617" MODIFIED="1588600559204" TEXT="System identifier?"/>
<node CREATED="1588596211483" ID="ID_1301204192" MODIFIED="1588596432975" TEXT="What&apos;s the relationship between Gaussian Process and RL? Is there any similarity between GP and KDE? Is GP the key to discretization a continuous problem into Q table?"/>
<node CREATED="1588468363862" ID="ID_1716223171" MODIFIED="1589206623976" TEXT="Refine the pre-defined model by using RL: Data-driven economic NMPC using reinforcement learning  &lt;a href=&quot;../papers/gros_tac2020.md&quot;&gt;gros&apos;s TAC2020&lt;/a&gt;"/>
<node CREATED="1588468382169" ID="ID_765878341" MODIFIED="1588596197621" TEXT="MPPI: Use neural network + bootstrapping as the optimizer for MPC: &lt;a href=&quot;../papers/william_icra2017.md&quot;&gt;Williams&apos; ICRA2017&lt;/a&gt;"/>
<node CREATED="1588600508311" ID="ID_1819802626" MODIFIED="1589206410115" TEXT="GP regression to correct the model Kabzan&apos;s RAL2019 &#x201c;learning-based model predictive control for autonomous racing&#x201d;"/>
</node>
</node>
<node CREATED="1588112792854" ID="ID_1240573700" MODIFIED="1588433937892" POSITION="left" TEXT="Type (slides: 1-linear-mpc, p59)">
<node CREATED="1588120934523" ID="ID_1442882100" MODIFIED="1588120943406" TEXT="linear MPC"/>
<node CREATED="1588102939746" ID="ID_691985331" MODIFIED="1589214082684" TEXT="NMPC (nonlinear)">
<node CREATED="1588346575895" ID="ID_1497450772" MODIFIED="1588346588132" TEXT="nonlinear: the system is nonlinear?"/>
<node CREATED="1589214085189" ID="ID_197390233" MODIFIED="1589214110042" TEXT="What is the  parameterized theta?"/>
<node CREATED="1588340077226" ID="ID_669444134" MODIFIED="1589214067979" TEXT="gros&apos;s TAC2020, use LQR(stage cost function) to replace value/Q func RL, Q-learning/TDAC to improve model"/>
</node>
<node CREATED="1588113395621" ID="ID_918028312" MODIFIED="1588347994955" TEXT="RMPC (Robust)"/>
<node CREATED="1588120945922" ID="ID_1331034311" MODIFIED="1588120952936" TEXT="stochastic MPC"/>
<node CREATED="1588120955460" ID="ID_644169511" MODIFIED="1588120965264" TEXT="distributed/decentralized MPC"/>
<node CREATED="1588102959550" ID="ID_627796200" MODIFIED="1588346220335" TEXT="economic MPC by rawling CDC2012">
<arrowlink DESTINATION="ID_627796200" ENDARROW="Default" ENDINCLINATION="0;0;" ID="Arrow_ID_1137083629" STARTARROW="None" STARTINCLINATION="0;0;"/>
<linktarget COLOR="#b0b0b0" DESTINATION="ID_627796200" ENDARROW="Default" ENDINCLINATION="0;0;" ID="Arrow_ID_1137083629" SOURCE="ID_627796200" STARTARROW="None" STARTINCLINATION="0;0;"/>
<node CREATED="1588346225698" ID="ID_1024323532" MODIFIED="1588346283980" TEXT="Difference: use operating cost directly as the stage cost (standard MPC use pre-designed stage cost)"/>
</node>
<node CREATED="1588120969487" ID="ID_1403243680" MODIFIED="1588120979954" TEXT="hybrid MPC">
<node CREATED="1588432709872" ID="ID_1400460368" MODIFIED="1588432755973" TEXT="Have a model selector, to control the switch affine system"/>
</node>
<node CREATED="1588120983215" ID="ID_1372208127" MODIFIED="1588120988828" TEXT="explicit MPC"/>
<node CREATED="1588120989311" ID="ID_1214699237" MODIFIED="1588120995903" TEXT="Solvers for MPC"/>
</node>
</node>
</map>
