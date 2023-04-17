import traceback

from datetime import datetime

from sqapi.datasource import get_datasource_plugin
from sqapi.geodata import GeodataAPI
from sqapi.ui import UIComponents


class GeodataCLI(GeodataAPI):
    _platform_id = None   # convenience property for storing selection

    def sync_deployment_ui(self, datasource_data, update_existing=False):
        try:
            # datasource = get_datasource(sqapi=self.sqapi, **datasource_data)
            DatasourceClass = get_datasource_plugin(datasource_type=datasource_data.get("datasource_type"))
            datasource = DatasourceClass(sqapi=self.sqapi, **datasource_data)
            campaign = UIComponents.select_object_list(datasource.list_campaigns(), title="Choose a CAMPAIGN:", list_format="{name}")
            deployment = UIComponents.select_object_list(datasource.list_deployments(campaign=campaign), title="Choose a DEPLOYMENT:", list_format="{name}")
            dataset = datasource.get_deployment_assets(campaign=campaign, deployment=deployment)
            return self.import_deployment(dataset, datasource, update_existing=update_existing)
            # print(json.dumps(dataset.dict(),indent=2))
        except Exception as e:
            traceback.print_exc()

    def sync_update_deployment_ui(self, datasource_data):
        return self.sync_deployment_ui(datasource_data, update_existing=True)

    def sync_campaign_ui(self, datasource_data, update_existing=False):
        started_at = datetime.now()
        # datasource = get_datasource(sqapi=self.sqapi, **datasource_data)
        DatasourceClass = get_datasource_plugin(datasource_type=datasource_data.get("datasource_type"))
        datasource = DatasourceClass(sqapi=self.sqapi, **datasource_data)
        campaign = UIComponents.select_object_list(datasource.list_campaigns(), title="Choose a CAMPAIGN:", list_format="{name}")
        results = self.import_campaign(datasource, campaign, update_existing=update_existing)
        self.send_email_results(results, title="Campaign: {datasource.name} > {campaign}".format(datasource=datasource, campaign=campaign.get("key")), started_at=started_at)
        return results

    def sync_update_campaign_ui(self, datasource_data):
        return self.sync_campaign_ui(datasource_data, update_existing=True)

    def sync_select_datasource_ui(self):
        # Select a platform
        platform = self.sqapi.select_resource_object_ui(resource="platform")

        # Choose a datasource for that platform and setup actions
        qs = {"q": {"filters": [{"name": "platform_id", "op": "eq", "val": platform.get("id")}]}, "results_per_page":1000}
        actions = [ #⤓	U+2913	Downwards Arrow to Bar
            dict(name="⤓ Sync ALL missing deployments and campaigns", callback=self.sync_datasource),
            dict(name="⤓ Sync WHOLE campaign (import any missing deployments)", callback=self.sync_campaign_ui),
            dict(name="⤓ Sync ONE missing deployment", callback=self.sync_deployment_ui),
            dict(name="✎ Reimport ONE deployment (can take a while)", callback=self.sync_update_deployment_ui),
            dict(name="✎ Reimport WHOLE campaign (update ALL deployments - takes long)", callback=self.sync_update_campaign_ui),
        ]
        return self.sqapi.select_resource_object_ui(resource="datasource", querystring_params=qs, actions=actions)

    def delete_deployment_ui(self, deployment):
        campaign = deployment.get("campaign")
        self.sqapi.confirm_delete(resource="deployment", id=deployment.get("id"), metadata=deployment)
        return self.list_deployments_ui(campaign)

    def delete_campaign_ui(self, campaign):
        self.sqapi.confirm_delete(resource="campaign", id=campaign.get("id"), metadata=campaign)
        return self.list_platform_campaigns_ui(None)   # reset to previous platform selection (in self._platform_id)

    def delete_platform_ui(self, platform):
        return self.sqapi.confirm_delete(resource="platform", id=platform.get("id"), metadata=platform)

    def edit_deployment_ui(self, deployment):
        raise NotImplementedError("Edit campaign not implemented yet")

    def edit_campaign_ui(self, campaign):
        raise NotImplementedError("Edit campaign not implemented yet")

    def edit_platform_ui(self, platform):
        raise NotImplementedError("Edit platform not implemented yet...")

    def list_deployments_ui(self, campaign):
        qs = dict(
            q=dict(
                filters=[dict(name="platform_id", op="eq", val=self._platform_id), dict(name="campaign_id", op="eq", val=campaign.get("id"))],
                order_by=[dict(field="created_at", direction="desc")]
            ),
            results_per_page=1000)
        actions = [
            dict(name="✎ Edit deployment", callback=self.edit_deployment_ui),
            dict(name="✖ Delete deployment", callback=self.delete_deployment_ui),
        ]
        return self.sqapi.select_resource_object_ui(resource="deployment", querystring_params=qs, actions=actions)

    def list_platform_campaigns_ui(self, platform):
        # set platform_id cache to store selection, otherwise just use previously stored value
        if platform is not None: self._platform_id = platform.get("id")
        qs = dict(
            q=dict(
                filters=[dict(name="deployments", op="any", val=dict(name="platform_id", op="eq", val=self._platform_id))],
                order_by=[dict(field="created_at", direction="desc")]
            ),
            results_per_page=1000)
        actions = [
            dict(name="☰ List campaign deployments", callback=self.list_deployments_ui),
            dict(name="✎ Edit campaign", callback=self.edit_campaign_ui),
            dict(name="✖ Delete campaign", callback=self.delete_campaign_ui),
        ]
        return self.sqapi.select_resource_object_ui(resource="campaign", querystring_params=qs, actions=actions)

    def list_platforms_ui(self):
        # Select a platform
        actions = [
            dict(name="☰ List platform campaigns", callback=self.list_platform_campaigns_ui),
            dict(name="✎ Edit platform", callback=self.edit_platform_ui),
            # dict(name="✖ Delete platform", callback=delete_platform),
        ]
        qs = {"results_per_page":1000, "q":{"order_by":[{"field":"name","direction":"asc"}]}}
        return self.sqapi.select_resource_object_ui(resource="platform", actions=actions, querystring_params=qs)