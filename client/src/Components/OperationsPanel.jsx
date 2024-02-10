import ArrowDropDownIcon from "@mui/icons-material/ArrowDropDown";
import ArrowRightIcon from "@mui/icons-material/ArrowRight";
import { CircularProgress } from "@mui/material";
import { styled } from "@mui/material/styles";
import { TreeView } from "@mui/x-tree-view";
import { TreeItem, treeItemClasses } from "@mui/x-tree-view/TreeItem";
import { useDispatch, useSelector } from "react-redux";
import { evaluateModel } from "../Slices/EvaluationSlice";

const StyledTreeItemRoot = styled(TreeItem)(({ theme }) => ({
  color: theme.palette.text.secondary,
  [`& .${treeItemClasses.content}`]: {
    color: "#3f3f46",
    borderRadius: theme.spacing(0.5),
    paddingRight: theme.spacing(1),
    paddingTop: theme.spacing(1),
    paddingBottom: theme.spacing(1),
    fontWeight: theme.typography.fontWeightBold,

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
  const { data, loading } = useSelector((state) => state.evaluation);
  const dispatch = useDispatch();

  const handleNodeSelect = async (b) => {
    setOperationID(b);
  };

  const handleModelCreate = async () => {
    dispatch(evaluateModel());
  };

  return (
    <div className="">
      <p className="font-semibold mb-4 text-2xl text-gray-800/80 text-center underline">
        Operations
      </p>
      <div className="text-center">
        <button
          className={`${
            !loading && "hover:text-gray-100 hover:bg-blue-500/90"
          } w-full text-white font-medium bg-blue-500 py-2 rounded`}
          disabled={loading}
          onClick={handleModelCreate}
        >
          {!loading ? (
            "Create Model"
          ) : (
            <CircularProgress color="inherit" size={"25px"} />
          )}
        </button>
      </div>
      <div className={`mt-3 ${loading && "pointer-events-none"}`}>
        {data && !loading && (
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
              {/* <StyledTreeItemRoot nodeId="r2" label="R-squared" /> */}
              <StyledTreeItemRoot
                nodeId="actualvspred"
                label="Actual vs Predicted"
              />
            </StyledTreeItemRoot>
            <StyledTreeItemRoot nodeId="structure" label="Structure Analysis" />
          </TreeView>
        )}
      </div>
    </div>
  );
};

export default OperationsPanel;
