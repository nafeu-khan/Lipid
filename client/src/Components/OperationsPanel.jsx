import ArrowDropDownIcon from "@mui/icons-material/ArrowDropDown";
import ArrowRightIcon from "@mui/icons-material/ArrowRight";
import { styled } from "@mui/material/styles";
import { TreeView } from "@mui/x-tree-view";
import { TreeItem, treeItemClasses } from "@mui/x-tree-view/TreeItem";
import { useSelector } from "react-redux";

const StyledTreeItemRoot = styled(TreeItem)(({ theme }) => ({
  color: theme.palette.text.secondary,
  [`& .${treeItemClasses.content}`]: {
    color: "#3f3f46",
    borderRadius: theme.spacing(0.5),
    paddingRight: theme.spacing(1),
    paddingTop: theme.spacing(1),
    paddingBottom: theme.spacing(1),
    fontWeight: theme.typography.fontWeightMedium,

    "&:hover": {
      backgroundColor: "#c7d2fe",
    },
    "&.Mui-focused, &.Mui-selected, &.Mui-selected.Mui-focused": {
      backgroundColor: `var(--tree-view-bg-color, ${"#6366f1"})`,
      color: "whitesmoke",
    },
    [`& .${treeItemClasses.label}`]: {
      fontWeight: "inherit",
      color: "inherit",
    },
  },
}));

const OperationsPanel = ({ setOperationID, operationID }) => {
  const loading = false;

  const handleNodeSelect = async (b) => {
    setOperationID(b);
  };

  return (
    <div className="">
      <p className="font-medium mb-1 text-lg text-gray-800/80 text-center underline">
        Operations
      </p>
      <div className={`mt-2 ${loading && "pointer-events-none"}`}>
        <TreeView
          aria-label="controlled"
          selected={operationID}
          onNodeSelect={(e, b) => handleNodeSelect(b)}
          defaultExpandIcon={<ArrowRightIcon />}
          defaultCollapseIcon={<ArrowDropDownIcon />}
        >
          <StyledTreeItemRoot nodeId="prediction" label="Prediction" />
          <StyledTreeItemRoot nodeId="evaluation" label="Evaluation">
            <StyledTreeItemRoot nodeId="loss" label="Train-Test Loss" />
            <StyledTreeItemRoot nodeId="r2" label="R-squared" />
            <StyledTreeItemRoot
              nodeId="actualvspred"
              label="Actual vs Predicted"
            />
          </StyledTreeItemRoot>
          <StyledTreeItemRoot nodeId="structure" label="Structure Analysis" />
        </TreeView>
      </div>
    </div>
  );
};

export default OperationsPanel;
