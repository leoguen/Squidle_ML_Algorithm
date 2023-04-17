from datetime import datetime
import time

from sqapi.datasource import get_datasource_plugin
from sqapi.api import SQAPIargparser, SQAPIBase
from sqapi.ui import lol2html

SQAPIargparser.add_argument("-e", '--email_results', action='append', default=[], type=str,
                       help="(optional) GEODATA: email sync results to user(s) with matching email address(es) (all must be registered email on server).")


class GeodataAPI:
    def __init__(self, sq=None, email_results=None, **sqapi_kwargs):
        self.sqapi = sq or SQAPIBase(**sqapi_kwargs)
        self.email_results = email_results #or self.sqapi.cliargs.email_results

    def import_deployment(self, dataset, datasource, update_existing=False):
        assert len(dataset.datafiles) == 1, \
            "Found {} datafiles for {}>{}. Importer currently only supports 1 datafile".format(
                len(dataset.datafiles), dataset.campaign.get("key"), dataset.deployment.get("key"))

        # Check campaign
        campaign, is_new = self.sqapi.get_create(data=dataset.get_campaign(), match_on=["key"], resource="campaign")

        # Check deployment
        deployment_data = dataset.get_deployment(campaign_id=campaign.get("id"), platform_id=datasource.platform.get("id"), datasource_id=datasource.id)
        # deployment_data["files"] = remote_files
        deployment, is_new_deployment = self.sqapi.get_create(data=deployment_data, match_on=["key", "campaign_id"], resource="deployment")
        deployment_id = deployment.get("id")

        # upload local files
        files = dataset.get_deployment_files(fileparams=dataset.deployment.get("fileparams", {}))
        for fdata in files:
            fileparams = fdata.pop("fileparams", None)      # get fileparams (if set, normally this would be done in `datasource.preprocess_deployment_assets()`
            fdata["deployment_id"] = deployment_id          # set `deployment_id`
            fobj, is_new_file = self.sqapi.get_create_file("deployment_file", match_on=["deployment_id", "name"], preprocess_file=datasource.preprocess_file, fileparams=fileparams, **fdata)

            save_media = False
            if is_new_deployment:   # if new deployment, run import on all files (this teats each file separately)
                save_media = dict(collection="deployment.media")
            elif update_existing:   # if not new deployment, but we're updating, then attempt to reimport (match on 'key' and 'deployment_id' (implicit through 'deployment.media' relation))
                save_media = dict(collection="deployment.media", update_existing=True, match_on=["key"], create_missing=True)
                if not is_new_file:  # If exists, but updating, then replace
                    print(" !! TODO: SHOULD PATCH  / UPDATE FILE - going to delete and re-post it for now")
                    self.sqapi.request(method="DELETE", resource="deployment_file/{id}".format(id=fobj.get("id")))
                    fobj = self.sqapi.create_file("deployment_file", preprocess_file=datasource.preprocess_file, fileparams=fileparams, **fdata)

            if save_media is not False:
                # f_ret = self.latest_deployment_file(deployment_id=deployment_id)
                params = dict(f=dict(operations=datasource.datafile_operations), save=save_media)
                rdata = self.sqapi.request("GET", resource="deployment_file/{id}/data", querystring_params=params, resource_params=dict(id=fobj.get("id")))
                self.sqapi.request("PATCH", resource="deployment/{id}", data_json={"is_valid": True}, resource_params=dict(id=deployment_id))
                result = dict(campaign=dataset.campaign.get("key"), deployment=dataset.deployment.get("key"),
                              message="{message} in {duration:.3f}s ({speed:.0f}/s, created: {created}, updated: {updated})".format(
                                  speed=(rdata.get("created",0)+rdata.get("updated",0))/rdata.get("duration",1), **rdata))
                print(" 💾 \033[1;32mSAVED! {campaign} > {deployment}\033[0m, {message}".format(**result))
                return result
            elif not deployment.get("is_valid"):
                raise ValueError("Deployment appears to exist but has `is_valid=False`. Try updating it to complete/correct the import process.")

        return None

    def import_campaign(self, datasource, campaign, update_existing=False):
        results = []
        for deployment in datasource.list_deployments(campaign=campaign):
            try:
                dataset = datasource.get_deployment_assets(campaign=campaign, deployment=deployment)
                r = self.import_deployment(dataset, datasource, update_existing=update_existing)
                if r is not None:
                    results.append(dict(status="imported", **r))
                else:
                    results.append(dict(status="skipped", campaign=campaign.get("key"), deployment=deployment.get("key"), message="No work done"))
            except Exception as e:
                r = dict(campaign=campaign.get("key"), deployment=deployment.get("key"), message="{}: {}".format(e.__class__.__name__, str(e)))
                print(" ❌ \033[1;31mERROR! {campaign} > {deployment}\033[0m, {message}".format(**r))
                results.append(dict(status="error", **r))
        return results

    def sync_datasource(self, datasource_data):
        results = []
        started_at = datetime.now()
        DatasourceClass = get_datasource_plugin(datasource_type=datasource_data.get("datasource_type"))
        datasource = DatasourceClass(sqapi=self.sqapi, **datasource_data)
        for c in datasource.list_campaigns():
            results += self.import_campaign(datasource, campaign=c)
        self.send_email_results(results, title="Datasource: {datasource[name]}".format(datasource=datasource_data), started_at=started_at)
        return results

    def send_email_results(self, results, title="", started_at=None):
        completed_at = datetime.now()

        if len(self.email_results)>0:
            print(" * Sending emails to: '{}'".format("'; '".join(self.email_results)))
            results_dict = dict(error=[], imported=[], skipped=[])
            results_count = {k:0 for k in results_dict.keys()}
            for i, r in enumerate(results):
                if r.get("status") in results_dict.keys():
                    results_count[r.get("status")] += 1
                    results_dict[r.get("status")].append(["{}.".format(results_count[r.get("status")]), "{campaign} > {deployment}: {message}".format(**r)])
            subject = "SQ+ Import Report | {title}".format(title=title)
            table_data = [["<b>SUMMARY</b>", "Imported: {}, skipped: {}, errors: {}".format(results_count.get("imported"),results_count.get("skipped"), results_count.get("error"))]]
            if started_at is not None:
                table_data.append(["<b>DURATION</b>", time.strftime('%H:%M:%S', time.gmtime((completed_at-started_at).total_seconds()))])
                table_data.append(["<b>STARTED</b>", str(started_at)])
            table_data.append(["<b>ENDED</b>", str(completed_at)])
            if results_count.get("imported") > 0:
                table_data.append(["<b>IMPORTED</b>", lol2html(results_dict.get('imported'))])
            if results_count.get("error") > 0:
                table_data.append(["<b>ERRORS</b>", lol2html(results_dict.get('error'))])

            message = lol2html(table_data)
            self.sqapi.send_user_email(subject, message, email_addresses=self.email_results)
