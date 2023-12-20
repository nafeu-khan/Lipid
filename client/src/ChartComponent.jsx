import * as am5 from "@amcharts/amcharts5";
import * as H from "@amcharts/amcharts5/hierarchy";
import am5themes_Animated from "@amcharts/amcharts5/themes/Animated";
import am5themes_Micro from "@amcharts/amcharts5/themes/Dataviz";
import am5themes_Responsive from "@amcharts/amcharts5/themes/Responsive";
import { useLayoutEffect } from "react";
import { graph_data } from "./utility";
// import "./App.css";

const data = JSON.parse(graph_data);

function ChartComponent({ active_lipid }) {
  useLayoutEffect(() => {
    var root = am5.Root.new("chartdiv");

    // root.setThemes([am5themes_Animated.new(root)]);
    root.setThemes([
      am5themes_Animated.new(root),
      am5themes_Micro.new(root),
      am5themes_Responsive.new(root),
    ]);

    var container = root.container.children.push(
      am5.Container.new(root, {
        width: am5.percent(100),
        height: am5.percent(100),
        layout: root.verticalLayout,
      })
    );

    var series = container.children.push(
      H.ForceDirected.new(root, {
        singleBranchOnly: false,
        downDepth: 1,
        initialDepth: 2,
        topDepth: 1,
        valueField: "value",
        categoryField: "name",
        childDataField: "children",
        nodePadding: 10,
        minRadius: 8,
        maxRadius: am5.percent(5),
        linkWithField: "linkWith",
        linkWithStrength: 1,
        idField: "name",
        manyBodyStrength: -30,
        centerStrength: 1,
      })
    );

    series.circles.template.setAll({
      fillOpacity: 0.8,
      strokeWidth: 2,
      strokeOpacity: 0.7,
    });

    series.links.template.setAll({
      strokeWidth: 1,
      strokeOpacity: 0.75,
    });

    series.links.template.states.create("active", {
      strokeWidth: 4,
      strokeOpacity: 1,
    });

    series.nodes.template.events.on("pointerover", function (ev) {
      am5.array.each(ev.target.dataItem.get("links"), function (link) {
        link.set("active", true);
      });
    });

    series.nodes.template.events.on("pointerout", function (ev) {
      am5.array.each(ev.target.dataItem.get("links"), function (link) {
        link.set("active", false);
      });
    });

    let graph_data = [];

    active_lipid.forEach((val) => {
      if (val.active) graph_data.push(data[val.name]);
    });

    series.data.setAll([
      {
        value: 0,
        children: graph_data,
      },
    ]);

    series.set("selectedDataItem", series.dataItems[0]);

    return () => {
      root.dispose();
    };
  }, [active_lipid]);

  return (
    <div
      id="chartdiv"
      className="flex-grow"
      style={{ width: "100%", height: "100%" }}
    ></div>
  );
}
export default ChartComponent;
