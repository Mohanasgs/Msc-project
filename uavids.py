import os, json, time, threading, queue, random, joblib, warnings, traceback, re
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import gradio as gr
import tracemalloc
import plotly.express as px

warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
warnings.filterwarnings("ignore", category=FutureWarning)

def _patch_numpy_for_old_pickles():
    try:
        if not hasattr(np.ufunc,"module"):
            try: np.ufunc.module="numpy"
            except Exception:
                try: setattr(np.ufunc,"module","numpy")
                except Exception: pass
        if getattr(np.ufunc,"__module__",None)!="numpy":
            try: np.ufunc.__module__="numpy"
            except Exception: pass
    except Exception: pass
_patch_numpy_for_old_pickles()

def _env_banner():
    try:
        import sklearn, sys
        return f"ENV » numpy={np.__version__} | sklearn={sklearn.__version__} | joblib={joblib.__version__} | py={sys.version.split()[0]}"
    except Exception: return ""

def _lime_available():
    try:
        from importlib.util import find_spec
        if find_spec("lime") is None: return False
        from lime import lime_tabular
        _ = lime_tabular.LimeTabularExplainer
        return True
    except Exception: return False

def _lime_version():
    try:
        from importlib.metadata import version
        return version("lime")
    except Exception: return "unknown"

LIME_AVAILABLE=False
try:
    import lime
    from lime import lime_tabular
    LIME_AVAILABLE=_lime_available()
    print(f"LIME available (v{_lime_version()})" if LIME_AVAILABLE else "LIME not available")
except Exception as e:
    LIME_AVAILABLE=False
    print("LIME not available:", repr(e))

BUNDLES_ROOT="/content/drive/MyDrive/maindata/output2/bundles"
DEFAULT_MERGED={
    "gcscase3":"/content/drive/MyDrive/maindata/gcscase3/merged_gcscase3.csv",
    "uavcase1":"/content/drive/MyDrive/maindata/uavcase1/merged_uavcase1.csv",
    "access":"/content/drive/MyDrive/maindata/access/merged_access_COMPLETE.csv",
}
ANALYSIS_OPTIONS=["gcscase3","uavcase1","access","realtime_synthetic"]

DISPLAY_TITLE={
    "deauthentication":"Deauthentication","bruteforce":"Bruteforce","gps jamming":"GPS Jamming","evil twin":"Evil Twin",
    "ddos":"DDoS","dos":"DoS","mitm":"MITM","udp flooding":"UDP Flooding","icmp flooding":"ICMP Flooding",
    "scanning":"Scanning","reconnaissance":"Reconnaissance","replay":"Replay","normal":"Normal","benign":"Benign"
}

def _norm(s):
    t=str(s).strip().lower().replace("_"," ").replace("-"," ")
    return re.sub(r"\s+"," ",t)

ALIAS_GCSC3={"fake landing":"deauthentication","fakelanding":"deauthentication"}
ALIAS_NONE={}

def canonicalize(name):
    if name is None: return None
    t=_norm(name)
    scope=ALIAS_GCSC3 if CTX.get("bundle")=="gcscase3" else ALIAS_NONE
    t=scope.get(t,t)
    return t

def display_name(canon):
    if canon is None: return None
    c=canonicalize(canon)
    return DISPLAY_TITLE.get(c,c.title())

CTX={"bundle":None,"obj":None,"analysis_mode":None,"lime_explainer":None,"lime_training_data":None}
STREAM_ACTIVE=False
STREAM_STATS={"events":0,"alerts":0,"start_time":None,"alerts_by_class":{},"risk_counts":{"High":0,"Medium":0,"Low":0},"analysis_mode":None}
ALERT_QUEUE=queue.Queue()
STREAM_RATE_HZ=25
STREAM_EVENTS=15000
PERFORMANCE_METRICS={"total_predictions":0,"total_prediction_time":0.0,"memory_usage_mb":0.0,"lime_explanation_time":0.0,"lime_explanations":0,"false_positives":0,"true_positives":0,"false_negatives":0,"true_negatives":0}
LIME_EXPLANATIONS_CACHE={}

THREAT_INTEL_RAW={
 "DDoS":{"cve":"CVE-2019-19781, CVE-2020-5902","mitre":"T1498 - Network Denial of Service","risk":"High","description":"Distributed Denial of Service attack"},
 "DoS":{"cve":"CVE-2019-11477, CVE-2020-10108","mitre":"T1499 - Endpoint Denial of Service","risk":"High","description":"Single-source DoS"},
 "MITM":{"cve":"CVE-2017-13077, CVE-2017-13078","mitre":"T1557 - Adversary-in-the-Middle","risk":"High","description":"Man-in-the-middle"},
 "Evil Twin":{"cve":"CVE-2019-15126","mitre":"T1200 - Hardware Additions","risk":"High","description":"Rogue AP"},
 "Deauthentication":{"cve":"CVE-2017-13077","mitre":"T1090 - Proxy","risk":"Medium","description":"Wi-Fi deauth"},
 "Bruteforce":{"cve":"CVE-2017-6074","mitre":"T1110 - Brute Force","risk":"Medium","description":"Password brute force"},
 "Replay":{"cve":"CVE-2008-5161","mitre":"T1557 - Adversary-in-the-Middle","risk":"Medium","description":"Packet replay"},
 "GPS Jamming":{"cve":"N/A","mitre":"T1565 - Data Manipulation","risk":"High","description":"GPS jamming"},
 "Normal":{"cve":"N/A","mitre":"N/A","risk":"None","description":"Legit traffic"},
 "Benign":{"cve":"N/A","mitre":"N/A","risk":"None","description":"Legit traffic"},
 "Scanning":{"cve":"N/A","mitre":"T1046 - Network Service Scanning","risk":"Low","description":"Recon scanning"},
 "Reconnaissance":{"cve":"N/A","mitre":"T1595 - Active Scanning","risk":"Low","description":"Active recon"},
 "UDP Flooding":{"cve":"CVE-2019-5599","mitre":"T1498 - Network Denial of Service","risk":"High","description":"UDP flood"},
 "ICMP Flooding":{"cve":"CVE-1999-128","mitre":"T1498 - Network Denial of Service","risk":"Medium","description":"ICMP flood"}
}
THREAT_INTELLIGENCE={canonicalize(k):v for k,v in THREAT_INTEL_RAW.items()}

ALERTS_CFG_DEFAULT={"high":0.92,"medium":0.85,"low":0.78}
CLASS_THRESH_DEFAULTS={"high":0.88,"medium":0.82,"low":0.78}
MARGIN_GAP=0.20
THRESH_BONUS=0.02

def safe_json_read(path, default=None):
    try:
        if os.path.exists(path):
            with open(path,"r") as f: return json.load(f)
    except Exception: pass
    return default

def list_bundles(root):
    if not os.path.isdir(root): return []
    out=[]
    try:
        for d in sorted(os.listdir(root)):
            p=os.path.join(root,d)
            if os.path.isdir(p) and os.path.exists(os.path.join(p,"model.pkl")) and os.path.exists(os.path.join(p,"label_encoder.pkl")) and os.path.exists(os.path.join(p,"feature_spec.json")):
                out.append(d)
    except Exception: pass
    return out

def thresholds_from_spec(spec):
    raw=spec.get("thresholds",{})
    classes=list(spec.get("class_names",[]))
    out={}
    for name in classes:
        can=canonicalize(name)
        disp=display_name(can)
        if disp in {"Normal","Benign"}: out[can]=9.9; continue
        base=(raw.get(disp) or raw.get(can) or 0.0)
        if disp in {"DDoS","DoS","MITM","GPS Jamming","Evil Twin","UDP Flooding"}: out[can]=max(float(base),CLASS_THRESH_DEFAULTS["high"])
        elif disp in {"Bruteforce","Deauthentication","Replay","ICMP Flooding"}: out[can]=max(float(base),CLASS_THRESH_DEFAULTS["medium"])
        else: out[can]=max(float(base),CLASS_THRESH_DEFAULTS["low"])
    return out

def prep_df_numeric(df, spec):
    feats=spec.get("feature_names",[])
    med=spec.get("feature_medians",{})
    for c in feats:
        if c not in df.columns: df[c]=np.nan
    X=df.reindex(columns=feats)
    X=X.apply(pd.to_numeric, errors="coerce").fillna(value=med)
    return X.astype(np.float32)

def severity_for(prob, thr, cfg):
    hi=float(cfg.get("high",0.90)); md=float(cfg.get("medium",0.80)); lo=float(cfg.get("low",0.70))
    if prob>=hi: return "HIGH"
    if prob>=md: return "MEDIUM"
    if prob>=max(lo,thr): return "LOW"
    return "INFO"

def _robust_load(path):
    try: return joblib.load(path)
    except Exception as e:
        msg=str(e)
        if "numpy.ufunc' object has no attribute 'module" in msg or "'numpy.ufunc' object has no attribute 'module'" in msg:
            import pickle
            with open(path,"rb") as f: return pickle.load(f)
        raise

def _ensure_model_loaded():
    if not CTX.get("obj"): raise RuntimeError("No analysis option selected. Load an option first.")
    obj=CTX["obj"]
    if "model" in obj and "le" in obj: return
    if not os.path.exists(obj["model_path"]): raise RuntimeError(f"Model file not found: {obj['model_path']}")
    if not os.path.exists(obj["le_path"]): raise RuntimeError(f"Label encoder file not found: {obj['le_path']}")
    _patch_numpy_for_old_pickles()
    obj["model"]=_robust_load(obj["model_path"])
    obj["le"]=_robust_load(obj["le_path"])
    if not hasattr(obj["le"],"classes_"): raise RuntimeError("Label encoder missing classes_")
    _setup_lime_explainer()

def _setup_lime_explainer():
    if not LIME_AVAILABLE: CTX["lime_explainer"]=None; return
    try:
        analysis_mode=CTX["analysis_mode"]
        training=None
        csv=DEFAULT_MERGED.get(analysis_mode)
        if csv and os.path.exists(csv):
            try:
                s=pd.read_csv(csv,nrows=4000,engine="python",on_bad_lines="skip",dtype=str)
                s=prep_df_numeric(s,CTX["obj"]["spec"]); training=s.values
            except Exception: training=None
        if training is None:
            feats=CTX["obj"]["spec"]["feature_names"]; med=CTX["obj"]["spec"]["feature_medians"]
            training=np.array([[med.get(f,0.0) for f in feats] for _ in range(2000)],dtype=np.float32)
        CTX["lime_training_data"]=training
        class_titles=[display_name(canonicalize(c)) for c in CTX["obj"]["le"].classes_]
        CTX["lime_explainer"]=lime_tabular.LimeTabularExplainer(training,feature_names=CTX["obj"]["spec"]["feature_names"],class_names=class_titles,mode="classification",discretize_continuous=True,random_state=42)
    except Exception: CTX["lime_explainer"]=None

def _lime_predict_wrapper(X_arr):
    feats=CTX["obj"]["spec"]["feature_names"]
    return CTX["obj"]["model"].predict_proba(pd.DataFrame(X_arr,columns=feats))

def _fast_explanation_from_importance(model, feat_names, top_k=5):
    try:
        imp=getattr(model,"feature_importances_",None)
        if imp is None or len(imp)!=len(feat_names): return "approx: top drivers unavailable"
        order=np.argsort(imp)[::-1][:top_k]
        return "approx: " + ", ".join([f"{feat_names[i]} (imp={int(imp[i])})" for i in order])
    except Exception: return "approx: explanation unavailable"

def _cache_lime_pairs(cls, pairs):
    can=canonicalize(cls)
    LIME_EXPLANATIONS_CACHE.setdefault(can,[]).append(list(pairs))
    if len(LIME_EXPLANATIONS_CACHE[can])>60:
        LIME_EXPLANATIONS_CACHE[can]=LIME_EXPLANATIONS_CACHE[can][-60:]

def get_lime_explanation(arr, k, cls, max_features=8):
    if not LIME_AVAILABLE or CTX.get("lime_explainer") is None:
        return _fast_explanation_from_importance(CTX["obj"]["model"], CTX["obj"]["spec"]["feature_names"])
    t0=time.time()
    try:
        exp=CTX["lime_explainer"].explain_instance(arr.flatten(), _lime_predict_wrapper, num_features=max_features, top_labels=1)
        pairs=exp.as_list(label=k); _cache_lime_pairs(cls,pairs)
        PERFORMANCE_METRICS["lime_explanation_time"]+=(time.time()-t0)
        PERFORMANCE_METRICS["lime_explanations"]+=1
        return "\n".join([f"- {r} ({'SUPPORTS' if w>0 else 'OPPOSES'}, |w|={abs(w):.3f})" for r,w in pairs])
    except Exception:
        return _fast_explanation_from_importance(CTX["obj"]["model"], CTX["obj"]["spec"]["feature_names"])

def _threshold_weighted_argmax(p, enc_classes, thresholds):
    can=[canonicalize(c) for c in enc_classes]
    w=np.array([p[i]/max(float(thresholds.get(can[i],0.5)),1e-6) for i in range(len(enc_classes))],dtype=np.float64)
    w=np.maximum(w,1e-12); w=w/np.sum(w)
    o=np.argsort(w)[::-1]; k=int(o[0]); k2=int(o[1]) if len(o)>1 else k
    return k,k2,float(w[k]),float(w[k2])

def _truth_is_attack(label_str):
    if label_str is None: return None
    s=_norm(label_str)
    if s in {"","unknown","nan"}: return None
    if s in {"normal","benign","legit","legitimate","benign traffic","normal traffic"}: return False
    if s in {"attack","malicious","abnormal"}: return True
    if s in {"0","false","neg","negative"}: return False
    if s in {"1","true","pos","positive"}: return True
    if re.fullmatch(r"[01]", s): return s=="1"
    return canonicalize(s) not in {"normal","benign"}

def predict_packet_row(packet_data, true_label=None, force_lime=False, obj=None):
    if obj is None: _ensure_model_loaded(); obj=CTX["obj"]
    spec=obj["spec"]; th=obj["thresholds"]; cfg=obj["alerts_cfg"]
    t0=time.time(); tracemalloc.start()
    df=pd.DataFrame([packet_data]) if isinstance(packet_data,dict) else pd.DataFrame([packet_data.to_dict()])
    X=prep_df_numeric(df,spec)
    probs=obj["model"].predict_proba(X); enc=list(obj["le"].classes_)
    p=probs[0]
    k,k2,adj_top,adj_runner=_threshold_weighted_argmax(p,enc,th)
    raw_pred=enc[k]; pred_can=canonicalize(raw_pred); pred_disp=display_name(pred_can)
    prob=float(p[k]); thr=float(th.get(pred_can,0.75))
    margin=adj_top-adj_runner
    sev=severity_for(prob,thr,cfg)
    alert=(pred_can not in {"normal","benign"}) and (prob>=(thr+THRESH_BONUS)) and (margin>=MARGIN_GAP) and (sev in ["HIGH","MEDIUM","LOW"])
    info=THREAT_INTELLIGENCE.get(pred_can,{"risk":"Medium","cve":"N/A","mitre":"N/A","description":""})
    lime_text=get_lime_explanation(X.values,k,pred_disp,8) if (alert or force_lime) else _fast_explanation_from_importance(obj["model"],spec["feature_names"])
    dt=time.time()-t0; mem=tracemalloc.get_traced_memory()[0]/1024/1024; tracemalloc.stop()
    PERFORMANCE_METRICS["total_predictions"]+=1
    PERFORMANCE_METRICS["total_prediction_time"]+=dt
    PERFORMANCE_METRICS["memory_usage_mb"]=max(PERFORMANCE_METRICS["memory_usage_mb"],mem)
    if true_label is not None:
        t=_truth_is_attack(true_label)
        if t is not None:
            pa=alert
            if pa and not t: PERFORMANCE_METRICS["false_positives"]+=1
            elif pa and t: PERFORMANCE_METRICS["true_positives"]+=1
            elif not pa and t: PERFORMANCE_METRICS["false_negatives"]+=1
            else: PERFORMANCE_METRICS["true_negatives"]+=1
    return pred_disp, prob, thr, margin, alert, sev, info, lime_text, dt*1000

def _warm_lime_cache(min_per_class=3, max_total=60):
    if not (LIME_AVAILABLE and CTX.get("lime_explainer") is not None and CTX.get("obj")): return
    try: _ensure_model_loaded()
    except Exception: return
    obj=CTX["obj"]; X_train=CTX.get("lime_training_data")
    if X_train is None or len(X_train)==0: return
    enc=obj["le"].classes_; want={canonicalize(c):0 for c in enc}; total=0; i=0
    while total<max_total and i<len(X_train):
        x=X_train[i]; probs=_lime_predict_wrapper(x.reshape(1,-1))[0]
        k=int(np.argmax(probs)); c=display_name(canonicalize(enc[k]))
        if want[canonicalize(enc[k])]<min_per_class:
            _=get_lime_explanation(x.reshape(1,-1),k,c,8); want[canonicalize(enc[k])]+=1; total+=1
        i+=1

def get_lime_feature_importance_plot():
    if not LIME_AVAILABLE: return None
    _warm_lime_cache()
    if not LIME_EXPLANATIONS_CACHE: return None
    rows=[]
    for attack_class, exps in LIME_EXPLANATIONS_CACHE.items():
        if attack_class in {"normal","benign"}: continue
        agg={}
        for pair_list in exps[-10:]:
            for rule,w in pair_list:
                feat=rule.split(' ')[0]
                agg.setdefault(feat,[]).append(abs(w))
        if agg:
            top=sorted(((f,float(np.mean(ws))) for f,ws in agg.items()), key=lambda x:x[1], reverse=True)[:5]
            for f,imp in top: rows.append({"Attack_Class":display_name(attack_class),"Feature":f,"Importance":imp})
    if not rows: return None
    df=pd.DataFrame(rows)
    fig=px.bar(df,x="Feature",y="Importance",color="Attack_Class",title="LIME Feature Importance by Attack Class",height=500)
    fig.update_layout(xaxis_tickangle=-45,margin=dict(l=20,r=20,t=60,b=20))
    return fig

def get_threshold_vs_lime_plot():
    if not CTX.get("obj"): return None
    th=CTX["obj"]["thresholds"]; rows=[]
    for cls,thr in th.items():
        if cls in {"normal","benign"}: continue
        risk=THREAT_INTELLIGENCE.get(cls,{}).get("risk","Medium")
        rows.append({"Attack_Class":display_name(cls),"Detection_Threshold":thr,"Risk_Level":risk})
    if not rows: return None
    df=pd.DataFrame(rows)
    fig=px.bar(df,x="Attack_Class",y="Detection_Threshold",color="Risk_Level",title="Detection Thresholds by Risk Level",height=400)
    fig.update_layout(xaxis_tickangle=-45,margin=dict(l=20,r=20,t=60,b=20))
    return fig

def load_bundle(option):
    if not option: return {"error":"No analysis option provided"}
    bundle_name="gcscase3" if option=="realtime_synthetic" else option
    bundle_dir=os.path.join(BUNDLES_ROOT,bundle_name)
    if not os.path.isdir(bundle_dir):
        return {"error":f"Bundle directory not found: {bundle_dir}. Available: {list_bundles(BUNDLES_ROOT)}"}
    req={"feature_spec.json":os.path.join(bundle_dir,"feature_spec.json"),
         "model.pkl":os.path.join(bundle_dir,"model.pkl"),
         "label_encoder.pkl":os.path.join(bundle_dir,"label_encoder.pkl")}
    for k,v in req.items():
        if not os.path.exists(v): return {"error":f"Missing {k} in {bundle_dir}"}
    spec=safe_json_read(req["feature_spec.json"],{})
    for f in ["class_names","feature_names","feature_medians"]:
        if f not in spec: return {"error":f"feature_spec.json missing field: {f}"}
    obj={"bundle_name":bundle_name,"analysis_mode":option,"bundle_dir":bundle_dir,"spec":spec,"alerts_cfg":ALERTS_CFG_DEFAULT,
         "model_path":req["model.pkl"],"le_path":req["label_encoder.pkl"],"thresholds":thresholds_from_spec(spec)}
    CTX["bundle"]=bundle_name; CTX["obj"]=obj; CTX["analysis_mode"]=option
    try: _ensure_model_loaded()
    except Exception as e: return {"error":f"Failed to load model: {e}"}
    global PERFORMANCE_METRICS, LIME_EXPLANATIONS_CACHE
    PERFORMANCE_METRICS={k:(0 if isinstance(v,(int,float)) else v) for k,v in PERFORMANCE_METRICS.items()}
    LIME_EXPLANATIONS_CACHE={}
    mode_desc="Synthetic Traffic" if option=="realtime_synthetic" else f"{option.upper()} Data"
    return {"ok":True,"msg":f"{mode_desc} loaded: {len(spec['class_names'])} classes, {len(spec['feature_names'])} features."}

def load_real_data_stream(option,max_rows=None):
    csv=DEFAULT_MERGED.get("gcscase3") if option=="realtime_synthetic" else DEFAULT_MERGED.get(option)
    if not csv or not os.path.exists(csv): return None
    try: return pd.read_csv(csv,nrows=max_rows,engine="python",on_bad_lines="skip",dtype=str)
    except Exception: return None

def generate_synthetic_packet(med):
    out={}
    for f,m in med.items():
        if m==0 or not np.isfinite(m): out[f]=abs(random.gauss(0.15,0.08))
        else:
            s=max(abs(m)*0.12,0.001); v=random.gauss(m,s)
            if any(k in f.lower() for k in ["count","size","len","bytes","packet"]): v=max(0.0,v)
            if random.random()<0.12: v=v*random.uniform(2.0,8.0)+random.uniform(4.0,40.0)
            out[f]=float(v)
    return out

def get_performance_stats():
    if PERFORMANCE_METRICS["total_predictions"]==0: return "No predictions made yet"
    avg=(PERFORMANCE_METRICS["total_prediction_time"]/PERFORMANCE_METRICS["total_predictions"])*1000
    lime_n=max(PERFORMANCE_METRICS["lime_exPLANATIONS"] if "lime_exPLANATIONS" in PERFORMANCE_METRICS else PERFORMANCE_METRICS["lime_explanations"],1)
    lime_ms=(PERFORMANCE_METRICS["lime_explanation_time"]/lime_n)*1000
    thrpt=PERFORMANCE_METRICS["total_predictions"]/max(PERFORMANCE_METRICS["total_prediction_time"],0.001)
    fp=PERFORMANCE_METRICS["false_positives"]; tn=PERFORMANCE_METRICS["true_negatives"]; fpr=fp/max(fp+tn,1)*100
    tp=PERFORMANCE_METRICS["true_positives"]; fn=PERFORMANCE_METRICS["false_negatives"]; prec=tp/max(tp+fp,1)*100; rec=tp/max(tp+fn,1)*100
    return (f"**LightGBM Performance**\n"
            f"- Avg Prediction: {avg:.2f} ms  |  Throughput: {thrpt:.1f}/s\n"
            f"- Max RSS: {PERFORMANCE_METRICS['memory_usage_mb']:.1f} MB  |  LIME cost: {lime_ms:.2f} ms/exp\n"
            f"- FPR: {fpr:.2f}%  |  Precision: {prec:.2f}%  |  Recall: {rec:.2f}%\n"
            f"- Total Predictions: {PERFORMANCE_METRICS['total_predictions']:,}")

def start_realtime_monitoring():
    global STREAM_ACTIVE, STREAM_STATS
    if not CTX.get("obj"): return "ERROR: No analysis option selected. Please select an option first.", pd.DataFrame(), "", ""
    try: _ensure_model_loaded()
    except Exception as e: return f"ERROR: {e}", pd.DataFrame(), "", ""
    if STREAM_ACTIVE: return "WARNING: Monitoring already running!", pd.DataFrame(), "", get_monitoring_stats()
    mode=CTX["obj"]["analysis_mode"]
    for fp in [os.path.join(CTX["obj"]["bundle_dir"],f"realtime_{mode}_alerts.csv"),os.path.join(CTX["obj"]["bundle_dir"],f"realtime_{mode}_enriched.csv")]:
        try:
            if os.path.exists(fp): os.remove(fp)
        except: pass
    STREAM_STATS={"events":0,"alerts":0,"start_time":time.time(),"alerts_by_class":{},"risk_counts":{"High":0,"Medium":0,"Low":0},"analysis_mode":mode}
    STREAM_ACTIVE=True
    threading.Thread(target=realtime_monitoring_loop,daemon=True).start()
    desc="Synthetic Traffic" if mode=="realtime_synthetic" else f"{mode.upper()} Data"
    return f"STARTED: Near-realtime monitoring with LightGBM{' + LIME' if LIME_AVAILABLE else ''}: {desc}", pd.DataFrame(), "", get_monitoring_stats()

def stop_realtime_monitoring():
    global STREAM_ACTIVE
    STREAM_ACTIVE=False
    mode=STREAM_STATS.get("analysis_mode","unknown")
    return f"STOPPED: Near-realtime monitoring of {mode} terminated", pd.DataFrame(), "", get_monitoring_stats()

def realtime_monitoring_loop():
    global STREAM_STATS, STREAM_ACTIVE
    try: _ensure_model_loaded(); obj=CTX["obj"]
    except Exception: STREAM_ACTIVE=False; return
    mode=obj["analysis_mode"]; tick=1.0/STREAM_RATE_HZ
    raw=[]; enr=[]; last=time.time()
    src=load_real_data_stream(mode,max_rows=STREAM_EVENTS); use_real=src is not None
    total=len(src) if use_real else STREAM_EVENTS
    for idx in range(total):
        if not STREAM_ACTIVE: break
        t0=time.time()
        try:
            true=None
            if use_real:
                row=src.iloc[idx]; packet=row
                for c in ['Label','label','attack_type','Attack_Type','class','Class']:
                    if c in row.index and pd.notna(row[c]): true=row[c]; break
            else:
                packet=generate_synthetic_packet(obj["spec"]["feature_medians"])
            cls,prob,thr,margin,alert,sev,info,lime_text,ms=predict_packet_row(packet,true,obj=obj)
            STREAM_STATS["events"]+=1
            if alert:
                STREAM_STATS["alerts"]+=1
                STREAM_STATS["alerts_by_class"][cls]=STREAM_STATS["alerts_by_class"].get(cls,0)+1
                risk=info.get("risk","Medium"); STREAM_STATS["risk_counts"][risk]=STREAM_STATS["risk_counts"].get(risk,0)+1
                ts=datetime.now(timezone.utc).isoformat(timespec="seconds")
                base={"timestamp":ts,"data_source":f"{mode}_{'real' if use_real else 'synthetic'}","sample_index":idx,"attack_type":cls,"probability":round(prob,4),"threshold":round(thr,3),"severity":sev,"risk_level":risk,"prediction_time_ms":round(ms,2),"true_label":str(true) if true is not None else "Unknown"}
                raw.append(base)
                enr.append({**base,"cve":info.get("cve","N/A"),"mitre_attack":info.get("mitre","N/A"),"description":info.get("description",""),"lime_explanation":lime_text})
                ALERT_QUEUE.put(enr[-1])
            if (time.time()-last)>1.0 or len(raw)>=50:
                if raw: save_alerts_to_csv(raw,f"realtime_{mode}_alerts.csv"); raw=[]
                if enr: save_alerts_to_csv(enr,f"realtime_{mode}_enriched.csv"); enr=[]
                last=time.time()
            el=time.time()-t0
            if el<tick: time.sleep(tick-el)
        except Exception: continue
    if raw: save_alerts_to_csv(raw,f"realtime_{mode}_alerts.csv")
    if enr: save_alerts_to_csv(enr,f"realtime_{mode}_enriched.csv")

def save_alerts_to_csv(rows, filename):
    if not rows or not CTX.get("obj"): return
    try:
        fp=os.path.join(CTX["obj"]["bundle_dir"],filename); os.makedirs(os.path.dirname(fp),exist_ok=True)
        df=pd.DataFrame(rows); exists=os.path.exists(fp)
        df.to_csv(fp,mode='a',header=not exists,index=False)
    except Exception: pass

def get_monitoring_stats():
    if not STREAM_STATS: return "No monitoring data available"
    ev=STREAM_STATS.get("events",0); al=STREAM_STATS.get("alerts",0)
    rt=time.time()-STREAM_STATS.get("start_time",time.time()) if STREAM_STATS.get("start_time") else 0
    mode=STREAM_STATS.get("analysis_mode","unknown"); rate=(al/max(ev,1))*100; eps=ev/max(rt,1)
    top=sorted(STREAM_STATS.get("alerts_by_class",{}).items(),key=lambda x:x[1],reverse=True)[:3]
    msg=(f"Data Source: {mode.upper()} Data\n"
         f"Monitoring Status: {'ACTIVE' if STREAM_ACTIVE else 'INACTIVE'}  Runtime: {rt:.1f} s\n"
         f"Processed: {ev} ({eps:.1f}/s)  Threats Detected: {al}  Rate: {rate:.2f}%\n")
    for i,(k,v) in enumerate(top,1): msg+=f"{i}. {k}: {v}\n"
    return msg

STATIC_TIME_BUDGET_SEC=900
LIME_MAX_PER_CHUNK=5
LIME_MAX_TOTAL=250

def _progress_html(pct):
    pct=max(0,min(100,pct))
    return f"""
    <div style='width: 100%; background: #111827; border-radius: 8px; height: 18px;'>
      <div style='width:{pct:.1f}%; height:100%; background: linear-gradient(90deg, #6366f1, #22c55e); border-radius: 8px;'></div>
    </div>
    <div style='font-size:12px; color:#4b5563; margin-top:4px;'>{pct:.1f}%</div>"""

def _synthetic_df(spec,n):
    med=spec.get("feature_medians",{}); feats=spec.get("feature_names",[])
    rows=[generate_synthetic_packet(med) for _ in range(int(n))]
    return pd.DataFrame(rows).reindex(columns=feats)

def run_static_analysis(use_default_csv, uploaded_file, top_n=20000, batch=4000, progress=gr.Progress(track_tqdm=True)):
    try:
        if not CTX.get("obj"):
            yield pd.DataFrame(),"ERROR: No analysis option selected.",_progress_html(0); return
        _ensure_model_loaded()
        obj=CTX["obj"]; spec=obj["spec"]; feats=spec.get("feature_names",[])
        use_synth=False
        if use_default_csv:
            fpath=DEFAULT_MERGED.get(obj["analysis_mode"])
            if not fpath or not os.path.exists(fpath): use_synth=True
        else:
            if uploaded_file is None: use_synth=True
            else:
                fpath=uploaded_file.name if hasattr(uploaded_file,"name") else str(uploaded_file)
                if not os.path.exists(fpath): use_synth=True
        run_dir=os.path.join(obj["bundle_dir"],"static_analysis",datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
        os.makedirs(run_dir,exist_ok=True)
        out_csv=os.path.join(run_dir,"threats_detected.csv")
        enc=np.array(obj["le"].classes_); th=obj["thresholds"]
        thr_idx=np.array([float(th.get(canonicalize(c),0.5)) for c in enc])
        total=thr=exp=0; t0=time.time(); preview=[]
        fast=_fast_explanation_from_importance(obj["model"],feats,5)
        def _emit(records):
            nonlocal thr, preview
            if not records: return
            df_out=pd.DataFrame(records); exists=os.path.exists(out_csv)
            df_out.to_csv(out_csv,mode='a',header=not exists,index=False)
            thr+=len(records); preview.extend(records)
            if len(preview)>200: preview=preview[-200:]

        if use_synth:
            rem=int(top_n); ch=int(batch)
            while rem>0 and (time.time()-t0)<STATIC_TIME_BUDGET_SEC:
                this=min(ch,rem)
                X=prep_df_numeric(_synthetic_df(spec,this),spec); n=len(X)
                probs=obj["model"].predict_proba(X); pred=np.argmax(probs,axis=1)
                pred_prob=probs[np.arange(n),pred]; runner=np.partition(probs,-2,axis=1)[:,-2]
                thr_vec=thr_idx[pred]
                idx=np.flatnonzero((pred_prob>=thr_vec+THRESH_BONUS) & ((pred_prob-runner)>=MARGIN_GAP))
                if idx.size:
                    k=min(LIME_MAX_PER_CHUNK,LIME_MAX_TOTAL-exp,idx.size)
                    pick=idx[np.argsort(-pred_prob[idx])[:k]] if k>0 else np.array([],dtype=int)
                    sel=set(pick.tolist()); recs=[]
                    for i in idx.tolist():
                        k_idx=int(pred[i]); cls=display_name(canonicalize(enc[k_idx])); p=float(pred_prob[i]); thrv=float(thr_vec[i])
                        sev=severity_for(p,thrv,obj["alerts_cfg"]); info=THREAT_INTELLIGENCE.get(canonicalize(enc[k_idx]),{})
                        expl=get_lime_explanation(X.iloc[i].values.reshape(1,-1),k_idx,cls,8) if (LIME_AVAILABLE and CTX.get("lime_explainer") is not None and i in sel) else fast
                        if i in sel: exp+=1
                        recs.append({"row_index_global": total+i,"severity":sev,"attack_type":cls,"probability":round(p,6),"threshold":round(thrv,3),"risk_level":info.get("risk","Medium"),"cve":info.get("cve","N/A"),"mitre_attack":info.get("mitre","N/A"),"description":info.get("description",""),"lime_explanation":expl,"prediction_time_ms":np.nan})
                    _emit(recs)
                total+=n; rem-=n; pct=min(100.0,(total/max(1,int(top_n)))*100.0)
                yield pd.DataFrame(preview) if preview else pd.DataFrame(), f"SYNTHETIC processing... {pct:.1f}% | rows {total:,} | alerts {thr} | LIME {exp} | output {out_csv}", _progress_html(pct)
        else:
            try: cols=pd.read_csv(fpath,nrows=1,engine="python",dtype=str).columns.tolist()
            except Exception: cols=None
            ycol=None
            for c in ['Label','label','attack_type','Attack_Type','class','Class']:
                if cols and c in cols: ycol=c; break
            usecols=[c for c in feats if (not cols or c in cols)]
            if ycol: usecols=sorted(set(usecols+[ycol]))
            if not usecols: usecols=None
            def _reader(engine):
                return pd.read_csv(fpath,chunksize=int(batch),usecols=usecols,engine=engine,on_bad_lines="skip",low_memory=False,dtype=str)
            try: reader=_reader("c")
            except Exception: reader=_reader("python")
            for chunk in reader:
                if total>=int(top_n) or (time.time()-t0)>STATIC_TIME_BUDGET_SEC: break
                if usecols is None:
                    keep=[c for c in feats+([ycol] if ycol else []) if c in chunk.columns]
                    if keep: chunk=chunk[keep]
                X=prep_df_numeric(chunk,spec); n=len(X)
                if n==0: continue
                probs=obj["model"].predict_proba(X); pred=np.argmax(probs,axis=1)
                pred_prob=probs[np.arange(n),pred]; runner=np.partition(probs,-2,axis=1)[:,-2]
                thr_vec=thr_idx[pred]; idx=np.flatnonzero((pred_prob>=thr_vec+THRESH_BONUS) & ((pred_prob-runner)>=MARGIN_GAP))
                ytrue=chunk[ycol].astype('string').fillna("unknown").str.lower() if ycol and (ycol in chunk.columns) else None
                if idx.size:
                    k=min(LIME_MAX_PER_CHUNK,LIME_MAX_TOTAL-exp,idx.size)
                    pick=idx[np.argsort(-pred_prob[idx])[:k]] if k>0 else np.array([],dtype=int)
                    sel=set(pick.tolist()); recs=[]
                    for i in idx.tolist():
                        k_idx=int(pred[i]); cls=display_name(canonicalize(enc[k_idx])); p=float(pred_prob[i]); thrv=float(thr_vec[i])
                        sev=severity_for(p,thrv,obj["alerts_cfg"]); info=THREAT_INTELLIGENCE.get(canonicalize(enc[k_idx]),{})
                        expl=get_lime_explanation(X.iloc[i].values.reshape(1,-1),k_idx,cls,8) if (LIME_AVAILABLE and CTX.get("lime_explainer") is not None and i in sel) else fast
                        if i in sel: exp+=1
                        rec={"row_index_global": total+i,"severity":sev,"attack_type":cls,"probability":round(p,6),"threshold":round(thrv,3),"risk_level":info.get("risk","Medium"),"cve":info.get("cve","N/A"),"mitre_attack":info.get("mitre","N/A"),"description":info.get("description",""),"lime_explanation":expl,"prediction_time_ms":np.nan}
                        if ytrue is not None: rec["true_label"]=str(ytrue.iloc[i])
                        recs.append(rec)
                    _emit(recs)
                total+=n; pct=min(100.0,(total/max(1,int(top_n)))*100.0)
                yield pd.DataFrame(preview) if preview else pd.DataFrame(), f"Processing... {pct:.1f}% | rows {total:,} | alerts {thr} | LIME {exp} | output {out_csv}", _progress_html(pct)
        dfp=pd.DataFrame(preview) if preview else pd.DataFrame()
        msg=(f"Complete{' with LIME explanations (sampled + fast)' if LIME_AVAILABLE else ''}: processed {min(total,int(top_n)):,} rows; threats: {thr}; LIME: {exp}. Output: {out_csv}")
        yield dfp,msg,_progress_html(100.0)
    except Exception:
        err=traceback.format_exc(limit=5)
        yield pd.DataFrame(), f"ERROR during static analysis:\n```\n{err}\n```", _progress_html(0)

title_suffix=" + LIME" if LIME_AVAILABLE else " (LIME disabled)"
custom_css=".gradio-container {max-width: 1220px !important} button {border-radius: 10px !important}"

with gr.Blocks(title=f"UAV IDS with LightGBM{title_suffix}", theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("## UAV IDS — LightGBM + LIME\n")
    with gr.Row():
        bundles_root=gr.Textbox(label="Bundles Root Directory", value=BUNDLES_ROOT)
        refresh_btn=gr.Button("Refresh Options")
        load_btn=gr.Button("Load Analysis Option", variant="primary")
    status_display=gr.Markdown()
    analysis_options=gr.Dropdown(label="Analysis Options", choices=list_bundles(BUNDLES_ROOT)+["realtime_synthetic"], value=None)

    with gr.Tabs():
        with gr.Tab("Detection Thresholds"):
            thresholds_table=gr.Dataframe(headers=["Attack_Type","Threshold"], interactive=False, wrap=True)
        with gr.Tab("Threat Intelligence"):
            threat_intel_table=gr.Dataframe(headers=["Attack_Type","Risk_Level","CVE","MITRE_Attack","Description"], interactive=False, wrap=True)
        with gr.Tab("LIME Explainability Visualizations"):
            with gr.Row():
                refresh_viz_btn=gr.Button("Refresh Visualizations", variant="primary")
            with gr.Row():
                lime_feature_plot=gr.Plot(label="LIME Feature Importance by Attack Class")
                lime_threshold_plot=gr.Plot(label="Detection Thresholds by Risk Level")
            refresh_viz_btn.click(lambda:(get_lime_feature_importance_plot(),get_threshold_vs_lime_plot()),outputs=[lime_feature_plot,lime_threshold_plot])
        with gr.Tab("Near-realtime Monitoring"):
            gr.Markdown("### Near-realtime LightGBM + LIME Threat Detection")
            with gr.Row():
                start_monitoring_btn=gr.Button("Start Trail", variant="primary")
                stop_monitoring_btn=gr.Button("Stop Trail")
                refresh_monitoring_btn=gr.Button("Refresh Data")
            monitoring_status=gr.Markdown("Ready.")
            alert_banner=gr.HTML("")
            with gr.Row():
                with gr.Column(scale=2):
                    realtime_alerts=gr.Dataframe(label="Live Threat Detections", interactive=False, wrap=True)
                with gr.Column(scale=1):
                    monitoring_stats=gr.Markdown("No monitoring data")
                    performance_stats=gr.Markdown("No performance data")
        with gr.Tab("Static Analysis"):
            gr.Markdown("### Analyze CSV or Synthetic Data")
            use_default_csv=gr.Checkbox(value=True,label="Use CSV dataset for selected option")
            default_csv_path=gr.Textbox(label="Default dataset path",interactive=False)
            upload_csv=gr.File(label="OR upload custom CSV file")
            with gr.Row():
                max_rows=gr.Number(label="Maximum rows to process", value=20000)
                batch_size=gr.Number(label="Processing batch size", value=4000)
            analyze_btn=gr.Button("Run LightGBM + LIME Analysis", variant="primary")
            analysis_results=gr.Dataframe(label="Detected Threats with Explanations", interactive=False, wrap=True)
            analysis_status=gr.Markdown()
            analysis_progress=gr.HTML(_progress_html(0))
        with gr.Tab("Performance Metrics"):
            performance_display=gr.Markdown("Make predictions or run near-realtime to see metrics")
            refresh_perf_btn=gr.Button("Refresh Performance Stats")
        with gr.Tab("System Information"):
            system_info=gr.Markdown()

    def cb_refresh_options(root):
        global BUNDLES_ROOT
        BUNDLES_ROOT=root.strip() or BUNDLES_ROOT
        bundles=list_bundles(BUNDLES_ROOT)
        options=bundles+["realtime_synthetic"]
        return gr.Dropdown(choices=options, value=(options[0] if options else None)), f"Available options: {options}"

    def cb_load_analysis_option(option):
        if not option:
            return ("Please select an analysis option.",pd.DataFrame(),pd.DataFrame(),"(none)",_env_banner())
        result=load_bundle(option)
        if "error" in result:
            return (f"ERROR: {result['error']}",pd.DataFrame(),pd.DataFrame(),"(none)",_env_banner())
        spec=CTX["obj"]["spec"]; th=CTX["obj"]["thresholds"]
        th_df=pd.DataFrame([{"Attack_Type":display_name(k),"Threshold":v} for k,v in sorted(th.items()) if k not in {"normal","benign"}])
        rows=[]
        for raw in spec.get("class_names",[]):
            can=canonicalize(raw); disp=display_name(can)
            intel=THREAT_INTELLIGENCE.get(can,{})
            rows.append({"Attack_Type":disp,"Risk_Level":intel.get("risk","Medium"),"CVE":intel.get("cve","N/A"),"MITRE_Attack":intel.get("mitre","N/A"),"Description":intel.get("description","")})
        ti_df=pd.DataFrame(rows)
        default_csv=DEFAULT_MERGED.get(option,"(none - will use synthetic data)")
        return (result["msg"], th_df, ti_df, default_csv, _env_banner())

    refresh_btn.click(cb_refresh_options, inputs=[bundles_root], outputs=[analysis_options, status_display])
    load_btn.click(cb_load_analysis_option, inputs=[analysis_options], outputs=[status_display, thresholds_table, threat_intel_table, default_csv_path, system_info])
    start_monitoring_btn.click(start_realtime_monitoring, outputs=[monitoring_status, realtime_alerts, alert_banner, monitoring_stats])
    stop_monitoring_btn.click(stop_realtime_monitoring, outputs=[monitoring_status, realtime_alerts, alert_banner, monitoring_stats])

    def refresh_realtime_data():
        if not CTX.get("obj"): return pd.DataFrame(), get_monitoring_stats(), "", get_performance_stats()
        mode=CTX["obj"]["analysis_mode"]
        alerts_file=os.path.join(CTX["obj"]["bundle_dir"],f"realtime_{mode}_enriched.csv")
        if os.path.exists(alerts_file):
            try:
                df=pd.read_csv(alerts_file)
                if len(df):
                    df["attack_type"]=df["attack_type"].map(lambda x: display_name(canonicalize(x)))
                recent=df.tail(100).iloc[::-1] if len(df) else pd.DataFrame()
                banner=""
                if len(df):
                    latest=df.iloc[-1]
                    src=latest.get('data_source',mode)
                    lime=str(latest.get('lime_explanation',''))[:160]
                    banner=f"""
                    <div style="background: linear-gradient(90deg, #1f2937, #111827); color: #fff; padding: 14px; border-radius: 10px; margin: 10px 0; border-left: 6px solid #6366f1;">
                        <div style="font-weight:600; color:#a5b4fc;">Explainable Security Alert — {src}</div>
                        <div style="opacity:.95; font-size:13px; margin-top:6px;"><b>{latest.get('attack_type','')}</b> | conf {float(latest.get('probability',0)):.3f} | risk {latest.get('risk_level','')} | MITRE {latest.get('mitre_attack','')}</div>
                        <div style="opacity:.85; font-size:12px; margin-top:6px;"><b>LIME:</b> {lime}</div>
                    </div>"""
                return recent, get_monitoring_stats(), banner, get_performance_stats()
            except Exception: pass
        return pd.DataFrame(), get_monitoring_stats(), "", get_performance_stats()

    refresh_monitoring_btn.click(refresh_realtime_data, outputs=[realtime_alerts, monitoring_stats, alert_banner, performance_stats])
    analyze_btn.click(run_static_analysis, inputs=[use_default_csv, upload_csv, max_rows, batch_size], outputs=[analysis_results, analysis_status, analysis_progress])
    refresh_perf_btn.click(get_performance_stats, outputs=[performance_display])

print(_env_banner())
demo.launch(server_name="0.0.0.0", server_port=7100, share=True, debug=True)
