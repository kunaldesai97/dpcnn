#
# Janky front-end to bring some sanity (?) to the litany of tools and switches
# in setting up, tearing down and validating your EKS cluster for working
# with k8s and istio.
#
# There is an intentional parallel between this makefile (eks.m for Minkube)
# and the corresponding file for Minikube (mk.m). This makefile makes extensive
# use of pseudo-target to automate the error-prone and tedious command-line
# needed to get your environment up. There are some deviations between the
# two due to irreconcilable differences between a private single-node
# cluster (Minikube) and a public cloud-based multi-node cluster (EKS).
#
# The intended approach to working with this makefile is to update select
# elements (body, id, IP, port, etc) as you progress through your workflow.
# Where possible, stodout outputs are tee into .out files for later review.
#

KC=kubectl
IC=istioctl

# these might need to change
NS=dpcnn
CLUSTERNAME=az-dpcnn
CTX=az-dpcnn
RSG=dpcnn
SUBS=6e512fd7-be7e-4897-9be8-3d05f5297b5e

connect:
	az account set --subscription $(SUBS)
	az aks get-credentials --resource-group $(RSG) --name $(CLUSTERNAME)

restart:
	az aks start --name $(CLUSTERNAME) --resource-group $(RSG) | tee aks-cluster.log

stop:
	az aks stop --name $(CLUSTERNAME) --resource-group $(RSG) | tee aks-cluster.log

delete:
	az aks delete --name $(CLUSTERNAME) --resource-group $(RSG) | tee aks-delete.log

show:
	az aks show --name $(CLUSTERNAME) --resource-group $(RSG) | tee aks-details.log


extern: showcontext
	$(KC) -n istio-system get service istio-ingressgateway

cd:
	$(KC) config use-context $(CTX)

# show svc across all namespaces
lsa: showcontext
	$(KC) get svc --all-namespaces

# show deploy and pods in current ns; svc of cmpt756e4 ns
ls: showcontext
	$(KC) get gw,deployments,pods
	$(KC) -n $(NS) get svc

# show containers across all pods
lsd:
	$(KC) get pods --all-namespaces -o=jsonpath='{range .items[*]}{"\n"}{.metadata.name}{":\t"}{range .spec.containers[*]}{.image}{", "}{end}{end}' | sort

# reinstate all the pieces of istio on a new cluster
# do this whenever you create/restart your cluster
# NB: You must rename the long context name down to $(CTX) before using this
reinstate:
	$(KC) config use-context $(CTX) | tee -a aks-reinstate.log
	$(KC) create ns $(NS) | tee -a aks-reinstate.log
	$(KC) config set-context $(CTX) --namespace=$(NS) | tee -a aks-reinstate.log
	$(KC) label ns $(NS) istio-injection=enabled | tee -a aks-reinstate.log
	$(IC) install --set profile=demo | tee -a aks-reinstate.log
	
showcontext:
	$(KC) config get-contexts
