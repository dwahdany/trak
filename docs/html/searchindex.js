Search.setIndex({docnames:["bert","clip","index","install","quickstart","slurm","trak"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":4,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.todo":2,"sphinx.ext.viewcode":1,sphinx:56},filenames:["bert.rst","clip.rst","index.rst","install.rst","quickstart.rst","slurm.rst","trak.rst"],objects:{"":[[6,0,0,"-","trak"]],"trak.gradient_computers":[[6,1,1,"","AbstractGradientComputer"],[6,1,1,"","FunctionalGradientComputer"],[6,1,1,"","IterativeGradientComputer"]],"trak.gradient_computers.AbstractGradientComputer":[[6,2,1,"","compute_loss_grad"],[6,2,1,"","compute_per_sample_grad"],[6,3,1,"","is_functional"],[6,2,1,"","load_model_params"]],"trak.gradient_computers.FunctionalGradientComputer":[[6,2,1,"","compute_loss_grad"],[6,2,1,"","compute_per_sample_grad"],[6,2,1,"","load_model_params"]],"trak.gradient_computers.IterativeGradientComputer":[[6,2,1,"","compute_loss_grad"],[6,2,1,"","compute_per_sample_grad"],[6,3,1,"","is_functional"],[6,2,1,"","load_model_params"]],"trak.modelout_functions":[[6,1,1,"","AbstractModelOutput"],[6,1,1,"","CLIPModelOutput"],[6,1,1,"","ImageClassificationModelOutput"],[6,1,1,"","IterImageClassificationModelOutput"],[6,1,1,"","TextClassificationModelOutput"]],"trak.modelout_functions.AbstractModelOutput":[[6,2,1,"","get_out_to_loss_grad"],[6,2,1,"","get_output"]],"trak.modelout_functions.CLIPModelOutput":[[6,2,1,"","get_embeddings"],[6,2,1,"","get_out_to_loss_grad"],[6,2,1,"","get_output"],[6,3,1,"","image_embeddings"],[6,3,1,"","num_computed_embeddings"],[6,3,1,"","sim_batch_size"],[6,3,1,"","text_embeddings"]],"trak.modelout_functions.ImageClassificationModelOutput":[[6,2,1,"","forward"],[6,2,1,"","get_out_to_loss_grad"],[6,2,1,"","get_output"]],"trak.modelout_functions.IterImageClassificationModelOutput":[[6,2,1,"","get_out_to_loss_grad"],[6,2,1,"","get_output"]],"trak.modelout_functions.TextClassificationModelOutput":[[6,2,1,"","forward"],[6,2,1,"","get_out_to_loss_grad"],[6,2,1,"","get_output"]],"trak.projectors":[[6,1,1,"","AbstractProjector"],[6,1,1,"","BasicProjector"],[6,1,1,"","BasicSingleBlockProjector"],[6,1,1,"","CudaProjector"],[6,1,1,"","ProjectionType"]],"trak.projectors.AbstractProjector":[[6,2,1,"","project"]],"trak.projectors.BasicProjector":[[6,2,1,"","generate_sketch_matrix"],[6,2,1,"","get_generator_states"],[6,2,1,"","project"]],"trak.projectors.BasicSingleBlockProjector":[[6,2,1,"","generate_sketch_matrix"],[6,2,1,"","project"]],"trak.projectors.CudaProjector":[[6,2,1,"","project"]],"trak.projectors.ProjectionType":[[6,3,1,"","normal"],[6,3,1,"","rademacher"]],"trak.savers":[[6,1,1,"","AbstractSaver"],[6,1,1,"","MmapSaver"],[6,4,1,"","ModelIDException"]],"trak.savers.AbstractSaver":[[6,2,1,"","clear_target_grad_count"],[6,2,1,"","del_grads"],[6,2,1,"","load_store"],[6,2,1,"","load_target_store"],[6,2,1,"","register_model_id"],[6,2,1,"","serialize_model_id_metadata"]],"trak.savers.MmapSaver":[[6,2,1,"","clear_target_grad_count"],[6,2,1,"","del_grads"],[6,2,1,"","init_store"],[6,2,1,"","load_store"],[6,2,1,"","load_target_store"],[6,2,1,"","register_model_id"],[6,2,1,"","save_scores"],[6,2,1,"","serialize_model_id_metadata"]],"trak.score_computers":[[6,1,1,"","AbstractScoreComputer"],[6,1,1,"","BasicScoreComputer"],[6,1,1,"","BasicSingleBlockScoreComputer"]],"trak.score_computers.AbstractScoreComputer":[[6,2,1,"","get_scores"],[6,2,1,"","get_x_xtx_inv"],[6,2,1,"","get_xtx"]],"trak.score_computers.BasicScoreComputer":[[6,2,1,"","get_scores"],[6,2,1,"","get_x_xtx_inv"],[6,2,1,"","get_xtx"]],"trak.score_computers.BasicSingleBlockScoreComputer":[[6,2,1,"","get_scores"],[6,2,1,"","get_x_xtx_inv"],[6,2,1,"","get_xtx"]],"trak.traker":[[6,1,1,"","TRAKer"]],"trak.traker.TRAKer":[[6,2,1,"","featurize"],[6,2,1,"","finalize_features"],[6,2,1,"","finalize_scores"],[6,2,1,"","init_projector"],[6,2,1,"","load_checkpoint"],[6,2,1,"","score"],[6,2,1,"","start_scoring_checkpoint"]],"trak.utils":[[6,5,1,"","get_num_params"],[6,5,1,"","get_params_dict"],[6,5,1,"","is_not_buffer"],[6,5,1,"","parameters_to_vector"],[6,5,1,"","vectorize_and_ignore_buffers"]],trak:[[6,0,0,"-","gradient_computers"],[6,0,0,"-","modelout_functions"],[6,0,0,"-","projectors"],[6,0,0,"-","savers"],[6,0,0,"-","score_computers"],[6,0,0,"-","traker"],[6,0,0,"-","utils"]]},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","attribute","Python attribute"],"4":["py","exception","Python exception"],"5":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:attribute","4":"py:exception","5":"py:function"},terms:{"0":[2,3,4,5,6],"000":4,"1":[2,3,4,5,6],"10":[3,4,5],"100":[4,6],"1000":2,"100000":6,"1024":6,"10_000":4,"11":3,"113":4,"12":3,"122":4,"125":4,"128":4,"14":5,"2":[2,4,6],"20":4,"200":6,"2000":4,"2048":6,"255":4,"256":4,"2nmcjaftdavyg0m":4,"3":[4,6],"300":6,"307":4,"4":[4,6],"4096":4,"4p73milxxafv4cm":4,"5":[3,4,6],"50":4,"50000":6,"50_000":6,"51":4,"512":4,"5865":4,"5e":4,"6":4,"64":4,"7":[3,4,6],"8":[4,5],"847":4,"8575":4,"8xa100":2,"9":[4,5],"961":4,"abstract":6,"break":6,"case":[3,4],"class":6,"default":[3,4,5,6],"do":[5,6],"enum":6,"export":3,"final":[4,5,6],"float":[4,6],"function":[4,6],"import":[4,5],"int":[4,5,6],"new":6,"return":[4,6],"static":6,"super":4,"true":[4,5,6],A:[4,6],Be:4,By:4,For:[3,4,5,6],If:[3,4,6],In:[4,6],It:6,On:[3,4],That:[4,5],The:[2,3,4,5,6],Then:4,These:6,To:[2,3,5],_:[4,6],__getitem__:4,__init__:4,__main__:5,__name__:5,_allow_featurizing_already_regist:6,_description_:6,_summary_:6,a100:[4,5,6],abc:6,ablat:4,about:[4,5,6],abov:[4,5,6],abstractgradientcomput:6,abstractmodeloutput:[4,6],abstractprojector:6,abstractsav:6,abstractscorecomput:6,accumul:6,achiev:5,across:[4,5,6],activ:2,ad:6,adapt:[2,4],adaptivemaxpool2d:4,add:[3,4],add_argu:5,addit:[4,6],addition:6,after:4,again:4,algorithm:2,all:[4,5,6],allow:6,almost:6,alon:6,along:6,alongsid:3,alreadi:[3,6],also:6,altern:4,alwai:4,amp:4,an:[2,3,4,6],analysi:6,analyt:6,analyz:4,ani:[4,6],anyth:6,api:[2,4],append:4,appli:[2,4,6],applic:6,apt:3,ar:[2,3,4,6],arang:4,architectur:4,arg:[5,6],argpars:5,argsort:4,argument:4,argumentpars:5,around:2,arrai:[4,5,6],art:2,as_tensor:4,assert:4,assertionerror:6,assign:4,associ:6,assum:[4,6],attempt:5,attention_mask:6,attribut:6,aug_se:4,autocast:4,autograd:6,avail:[3,6],avg:4,avoid:6,axi:4,b:6,backward:[4,6],bad:4,bar:4,bare:6,base:[2,6],bash:5,basicprojector:6,basicscorecomput:6,basicsingleblockprojector:6,basicsingleblockscorecomput:6,batch:[4,5,6],batch_siz:[4,6],batchnorm2d:4,becaus:4,been:[3,4],befor:[3,6],behavior:6,behaviour:6,behind:5,below:[3,4,5],bert:2,beton:4,beton_path:4,beton_url:4,better:4,bia:4,big:6,bigger:4,bin:5,bit:4,block:6,block_siz:6,bone:6,bool:6,both:6,bracket:3,bring:4,buffer:6,c:[3,6],call:[5,6],can:[2,3,4,5,6],cannot:6,capabl:6,caption:6,care:4,cat:4,cc:3,cdot:6,chang:2,channels_in:4,channels_last:4,channels_out:4,check:[2,3,4,5,6],checkpoint:[5,6],choic:6,chosen:6,ckpt:[4,5],ckpt_:5,ckpt_file:4,classif:[4,6],clean:4,clear_target_grad_count:6,clip:[2,4,6],clipmodeloutput:6,close:4,cluster:3,code:[2,4,5],com:[3,4,6],combin:[4,5],come:[0,1,2,3],command:[3,5],commit:6,common:4,compani:3,compat:[3,6],compil:3,comput:[5,6],compute_loss_grad:6,compute_per_sample_grad:6,conceptu:4,conda:3,confid:3,configur:3,confus:6,consid:6,consist:4,construct_rn9:4,contain:[2,3,4],context:6,contrast:[4,6],conv2d:4,conv_bn:4,convert:4,copi:[4,5],correct:[4,6],correctli:3,correl:4,count:6,cpu:[4,5],creat:[4,6],crossentropyloss:4,cuda:[2,4,6],cuda_max_dim_s:6,cudaprojector:6,cudatoolkit:3,curiou:4,current:5,current_:6,custom:[2,3,4,6],cutout:4,cxx:3,cyclic:4,data:[2,6],datamodel:6,dataset:4,decod:4,def:[4,5],defin:4,definit:5,del_grad:6,delet:6,depend:3,deriv:6,desc:4,design:[2,5,6],desir:6,detail:[2,4,5,6],determin:6,dev:3,develop:[2,6],devic:[4,6],diagon:6,dict:6,dictionari:6,did:4,differ:6,dimens:[4,6],dimension:[4,6],dir:6,disk:6,distribut:5,dl:4,doc:[3,5,6],document:3,doe:6,doesn:6,dog:4,don:[5,6],done:3,doubt:3,download:[3,4],drop_last:4,dropbox:4,ds_train:5,dtype:6,dure:[4,6],e:[2,3,4,5,6],each:[4,5,6],eas:[2,5],easi:2,effect:2,effici:2,either:[4,6],ell:6,embed:6,embedding_dim:6,enabl:[3,6],encourag:4,end:5,entir:[2,4,6],entri:[4,6],enumer:[4,6],ep:4,epoch:4,eq:6,equival:6,error:[4,6],escap:3,etc:[4,6],eval:4,eval_correl:4,even:6,everi:6,everyth:5,exact:4,exampl:[3,4,5,6],except:6,exist:[3,6],exist_ok:4,exmapl:5,exp:6,exp_nam:6,explain:4,ext:6,extend:4,f:[4,5,6],fact:4,fals:[4,6],faq:2,fast:[2,3],faster:6,feasibl:6,featur:[5,6],featurize_and_scor:5,feel:[3,6],few:2,ffcv:4,field:4,fifteen:4,figsiz:4,figur:4,file:[5,6],fill:4,finalize_featur:[4,5,6],finalize_scor:[4,5,6],find:5,fine:4,first:4,fix:6,flatten:[4,6],flexibl:2,float16:6,float32:[4,6],follow:[2,3,4,6],footprint:6,forbidden:6,forg:3,forward:[4,6],found:3,frac:6,free:6,from:[4,5,6],func:6,func_model:6,functionalgradientcomput:6,functorch:6,g:[2,3,4,5,6],gather:5,gener:6,generate_sketch_matrix:6,generator_st:6,get:[2,3,5],get_dataload:4,get_embed:6,get_generator_st:6,get_num_param:6,get_out_to_loss_grad:6,get_output:6,get_params_dict:6,get_scor:6,get_x_xtx_inv:6,get_xtx:6,getter:6,github:4,given:[4,6],go:[4,5],good:6,got:4,gpu:[4,5,6],grad:6,grad_dim:6,grad_out_to_loss:6,grad_w0:6,grad_w1:6,grad_wi:6,grad_wp:6,gradient:[2,4,6],gradientcomput:6,gradscal:4,gre:5,group:4,guid:[2,3],ha:[3,4,6],handl:6,happen:3,harmon:6,hash:4,have:[3,4,5,6],help:5,henc:6,here:[3,4,5,6],heta:6,highli:5,highlight:5,histogram:4,host:3,hour:2,how:[2,4,6],html:[3,6],http:[3,4,6],hyperparamet:6,i:[3,4,6],id:[5,6],idea:5,ight:6,im:4,imag:[4,6],image_classif:[4,5],image_embed:6,image_pipelin:4,imageclassificationmodeloutput:6,implement:6,imshow:4,includ:6,incompat:3,ind:[4,6],index:[2,3,4],indic:[4,6],ineffici:6,influenc:6,init:[4,6],init_projector:6,init_stor:6,initi:[4,6],initla:4,inplac:4,input:6,input_id:6,insid:4,instanc:[4,5],instead:[4,6],instruct:3,intdecod:4,integr:5,interfac:4,intermedi:[5,6],intern:6,interp:4,invok:6,involv:4,is_funct:6,is_not_buff:6,issu:4,iter:[4,6],iterativegradientcomput:6,iterimageclassificationmodeloutput:6,iters_per_epoch:4,its:[4,6],j:4,jl:[3,6],job:5,johnson:6,joinpath:4,just:[2,4],kernel:[2,3],kernel_s:4,know:4,kwarg:6,lab:4,label:[4,6],label_pipelin:4,label_smooth:4,lambdalr:4,languag:[4,6],larger:6,last:6,later:4,learn:[2,4,5,6],left:[5,6],len:[4,5],length:4,less:2,let:[3,4,6],leverag:[5,6],like:[5,6],lindenstrauss:6,line:[2,4,5],linear:4,link:6,list:[3,4,6],ll:[3,4,5,6],load:[3,4,5,6],load_checkpoint:[4,5,6],load_from_save_dir:6,load_model_param:6,load_state_dict:6,load_stor:6,load_target_stor:6,loader:[4,6],loader_target:4,loader_top_scor:4,loader_train:[4,5],loader_v:5,log:[3,6],logit:6,look:[4,5],loss:[4,6],loss_fn:4,lot:2,lr:4,lr_peak_epoch:4,lr_schedul:4,m:3,machin:3,mai:[2,3,5],main:5,maintain:3,make:[3,4,6],make_functional_with_buff:6,makedir:4,mani:[4,6],manner:6,map:[4,6],map_loc:4,margin:[4,6],margins_path:4,margins_url:4,mask:4,masks_path:4,masks_url:4,match:[3,4],math:6,matmul:6,matplotlib:4,matric:6,matrix:6,max:4,maximum:6,maxpool2d:4,mayb:4,mean:[4,5],memmap:6,memmori:4,memori:[4,6],memory_format:4,metadata:[4,6],method:[2,4,5,6],might:[3,4],min:4,minim:[5,6],minut:4,mmap:6,mmap_mod:4,mmapsav:6,modal:2,mode:6,model:[5,6],model_count:4,model_id:[4,5,6],model_out_to_loss:6,model_out_to_loss_grad:6,modelidexcept:6,modelout_fn:6,modeloutput:6,modifi:6,modul:[2,3,4],modulefil:3,momentum:4,more:[2,4,5,6],most:[3,4,6],much:6,mul:4,multiclass:6,multimod:6,multipl:[4,5],must:6,n:[4,6],nabla_:6,name:[5,6],natur:4,need:[3,4,5,6],nn:[4,6],node:[2,5],non_block:4,none:[4,6],normal:[4,6],notat:6,note:[3,4,5,6],notebook:6,now:[4,5],np:4,npy:[4,6],num:4,num_class:4,num_computed_embed:6,num_param:6,num_sampl:[4,6],num_target:[4,6],num_work:4,number:6,numpi:[4,6],nvidia:[3,6],object:[4,6],off:4,often:5,old:6,onc:4,one:[3,4,6],onli:[2,4,5,6],oom:6,open:4,oper:[3,4],opt:4,optim:4,option:[3,5,6],order:4,orderopt:4,org:6,os:4,other:3,otherwis:6,our:[2,3,4,5,6],out:[4,5,6],out_to_loss:6,outdat:3,output:6,over:4,overal:[4,5],overrid:[4,6],own:4,p:[4,6],packag:[2,3],pad:4,pair:6,paper:[2,4,5,6],parallel:4,paralleliz:5,paramet:[4,6],parameters_to_vector:6,params_dict:6,parse_arg:5,parser:5,part:6,partial:6,particular:[4,6],pass:[4,5,6],path:[4,5,6],pathlib:[4,6],per:[4,5,6],per_sample_grad:6,perform:[4,6],permut:4,peski:6,phi:6,phi_c:6,pick:3,pip:[2,3],pipelin:4,pleas:6,plese:3,plt:4,point:4,popul:[5,6],post:[4,5],power:5,ppa:3,pr:6,pred:4,predict:6,preinstal:3,prepar:6,preprocess_fn_img:6,preprocess_fn_txt:6,principl:2,print:4,prioriti:5,probabl:6,problem:[5,6],proce:3,process:[4,5],produc:[4,6],progress:4,proj_dim:[4,6],proj_typ:6,project:[3,6],projectiontyp:6,propos:6,provid:[4,6],ps:4,psi:6,pt:[4,5],purpos:[4,6],py:[3,5],pypi:3,pyplot:4,pytest:3,python:[3,5],pytorch:[2,6],q:6,qnli:2,quick:3,quickli:[3,4],quickstart:2,r:[3,4,6],rademach:6,rais:6,random:6,randomhorizontalflip:4,randomtransl:4,rang:[4,6],re:[4,6],read:6,readi:4,reason:6,receiv:4,recommend:6,record:6,reduct:[4,6],refer:[3,4],regard:3,regist:[4,6],register_model_id:6,relat:6,relu:4,remain:4,repositori:3,requir:[3,5,6],reset:6,reshap:6,residu:4,resnet18:6,resnet9:4,resnet:4,respect:[5,6],result:[4,5,6],reweight:[4,6],rglob:4,robust:4,root:3,roughli:4,rs:4,run:[3,4,5,6],rune:3,s:[2,3,4,5,6],same:[3,4,5,6],sampl:[4,6],save:6,save_dir:[4,5,6],save_scor:6,sbatch:5,scalar:6,scale:4,scaler:4,schedul:[4,5],scipi:4,score:6,scorecomput:6,scorer:4,script:[4,5],sd_:4,section:[3,4,6],see:[2,3,4,5,6],seed:[4,6],self:[4,6],sequenti:[4,6],serial:6,serialize_model_id_metadata:6,set:6,set_to_non:4,setter:6,setup:3,sgd:4,shape:[4,6],share:3,ship:3,should:[4,5,6],should_aug:4,shouldn:6,show:[3,4],shuffl:4,signatur:6,significantli:6,sim_batch_s:6,similar:[4,6],simpl:[5,6],simplergbimagedecod:4,simpli:6,simplic:2,simul:6,simulated_batch_s:6,sinc:6,singl:4,size:[4,6],sketch:5,skip:5,slightli:6,slower:6,slurm:4,slurm_array_task_id:5,small:6,snipeet:4,snippet:4,so:[4,6],softmax:6,some:[3,4,5],someth:6,soon:[0,1],sourc:6,spearmanr:4,specifi:[4,5,6],speed:2,split:4,squar:3,squeez:4,stabl:6,standard:3,start:[3,4,6],start_scoring_checkpoint:[4,5,6],stat:4,state:[2,6],state_dict:[4,6],std:4,step:[3,4,6],still:2,store:[4,6],str:[4,6],stride:4,string:4,strongli:6,structur:6,style:4,subclass:6,submit:5,subset:4,sudo:3,sum_:6,supervis:4,support:4,sure:3,surprisingli:4,sv:3,system:[3,5],t:[5,6],take:[2,4,6],target:6,target_grad:6,task:[2,4,5,6],tc3r3c3kgna2h27:4,temperatur:6,tensor:[4,6],term:[4,6],termin:[3,5],test:[3,6],test_rademach:3,text:[4,6],text_embed:6,textclassificationmodeloutput:6,than:[2,6],thei:3,them:[4,6],theta:6,thi:[2,3,4,5,6],thing:[2,5],think:[5,6],three:[5,6],thu:6,ti:[4,6],time:[3,4],tip:[3,4],tmp_path:4,todevic:4,todo:[4,6],togeth:6,token_type_id:6,tookit:3,toolchain:3,top:[4,6],torch:[4,5,6],torchvis:4,total:4,totensor:4,totorchimag:4,tqdm:4,track:4,train:[5,6],train_set_s:[4,5,6],trak_result:[4,6],traker:[2,3,5],transform:[4,6],triangl:4,trick:4,tupl:[4,6],tutori:4,twenti:4,two:4,type:[4,5,6],ubuntu:3,under:2,understand:6,union:6,uniqu:6,unless:6,unvers:3,up:5,updat:[3,4,6],us:[2,3,4,5,6],usual:3,uuid:6,val:4,val_ind:4,val_margin:4,valid:4,valu:[4,6],vector:6,vectorize_and_ignore_buff:6,veri:6,version:[2,3,6],via:6,view:[4,6],vision:6,vmap:6,wa:5,wai:[3,4,5],want:6,we:[2,4,5,6],weight:[4,6],weight_decai:4,well:6,wget:4,what:[3,4,5,6],when:[3,4,5,6],where:6,whether:6,which:[2,3,6],why:5,wise:6,wish:4,within:4,without:[3,6],work:4,would:5,write:[4,5,6],www:4,x:[2,4,6],xtx:6,y:6,ye:6,you:[2,3,4,5,6],your:[3,4,6],z:6,z_i:6,zero_grad:4,zn7jsp2rl09e0fh:4,zsh:3},titles:["Add  a <code class=\"code docutils literal notranslate\"><span class=\"pre\">task</span></code> to <code class=\"code docutils literal notranslate\"><span class=\"pre\">TRAKer</span></code> (subclassing <code class=\"code docutils literal notranslate\"><span class=\"pre\">ModelOutput</span></code>) \u2014 BERT-base","Add  a <code class=\"code docutils literal notranslate\"><span class=\"pre\">task</span></code> to <code class=\"code docutils literal notranslate\"><span class=\"pre\">TRAKer</span></code> (subclassing <code class=\"code docutils literal notranslate\"><span class=\"pre\">ModelOutput</span></code>) \u2014 CLIP","TRAK: Attributing Model Behavior at Scale","Installation FAQs","Quickstart \u2014 get <code class=\"code docutils literal notranslate\"><span class=\"pre\">TRAK</span></code> scores for <code class=\"code docutils literal notranslate\"><span class=\"pre\">CIFAR</span></code>","Parallelize <code class=\"code docutils literal notranslate\"><span class=\"pre\">TRAK</span></code> scoring with <code class=\"code docutils literal notranslate\"><span class=\"pre\">SLURM</span></code>","API reference"],titleterms:{"class":4,A:3,add:[0,1],api:6,attribut:[2,4],base:0,behavior:2,bert:0,checkpoint:4,cifar:4,clip:1,comput:4,content:[2,6],counterfactu:4,cuda:3,data:4,evalu:4,extra:4,faq:3,featur:4,gcc:3,get:4,gradient_comput:6,how:3,indic:2,instal:[2,3],misc:3,model:[2,4],modelout_funct:6,modeloutput:[0,1],modul:6,nvcc:3,overview:2,parallel:5,projector:6,q:3,quickstart:4,refer:6,save:4,saver:6,scale:2,score:[4,5],score_comput:6,set:4,slurm:5,subclass:[0,1],tabl:2,target:4,task:[0,1],toolkit:3,train:4,trak:[2,3,4,5,6],traker:[0,1,4,6],up:4,util:6,verifi:3,visual:4,work:3}})