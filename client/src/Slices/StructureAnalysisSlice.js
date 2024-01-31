import { createAsyncThunk, createSlice } from "@reduxjs/toolkit";
import { graph_data } from "../utility";

const graphData = JSON.parse(graph_data);
const URL = "http://127.0.0.1:8000";

export const getMoleculeStructure = createAsyncThunk(
  "lipid/getMoleculeStructure",
  async (mol_name, { rejectWithValue }) => {
    try {
      const res = await fetch(`${URL}/edge_pred/`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          mol_name,
        }),
      });
      const data = await res.json();

      return { data, mol_name };
    } catch (error) {
      return rejectWithValue(mol_name);
    }
  }
);

export const getPredictions = createAsyncThunk(
  "lipid/getPredictions",
  async (molecule, { rejectWithValue }) => {
    try {
      // TODO: update body if working with multiple lipid component. Below code only work for single component
      const body = {
        issingle: molecule.length === 1,
        lipid_name: molecule[0].name,
      };
      const res = await fetch(`${URL}/prediction/`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(body),
      });
      const data = await res.json();

      return data;
      // return { data, mol_name };
    } catch (error) {
      return rejectWithValue(error);
    }
  }
);

const initialState = {
  lipid: [{name: '', percentage: 100}],
  type: "single",
  loading: false,
  data: undefined,
  showTable: false,
};

export const structureAnalysis = createSlice({
  name: "lipid",
  initialState,
  reducers: {
    changeActiveLipid: (state, { payload }) => {
      state.lipid = payload;
    },
    changeNumOfComp: (state, { payload }) => {
      state.type = payload;
    },
    changeShowTable: (state) => {
      state.showTable = !state.showTable;
    },
  },
  extraReducers: (builder) => {
    builder.addCase(getMoleculeStructure.fulfilled, (state, { payload }) => {
      const { linkWith, ...rest } = payload.data.predicted_edge;
      state.data = {
        actual: { [payload.mol_name]: graphData[payload.mol_name] },
        predicted: { [payload.mol_name]: rest },
        edge_table: payload.data.edge_data,
      };
      state.loading = false;
    });
    builder.addCase(getMoleculeStructure.pending, (state) => {
      state.loading = true;
    });
    builder.addCase(getMoleculeStructure.rejected, (state, { payload }) => {
      state.data = { actual: undefined, predicted: undefined };
      if (graphData[payload]) {
        state.data = {
          ...state.data,
          actual: { [payload]: graphData[payload] },
        };
      }
      state.loading = false;
    });
    builder.addCase(getPredictions.pending, (state) => {
      state.loading = true;
    });
    builder.addCase(getPredictions.fulfilled, (state, { payload }) => {
      // TODO: change pred value while working with multiple component

      state.data = {
        pred: payload.pred[0][0],
        loss: { graph: payload.graph1, table: JSON.parse(payload.loss_df) },
        r2: { graph: payload.graph2, table: JSON.parse(payload.r2_df) },
        actualvspred: JSON.parse(payload.actualvspred),
      };
      state.loading = false;
    });
    builder.addCase(getPredictions.rejected, (state) => {
      state.data = {};
      state.loading = false;
    });
  },
});

// Action creators are generated for each case reducer function
export const {
  changeActiveLipid,
  changeNumOfComp,
  changeShowTable,
} = structureAnalysis.actions;

export default structureAnalysis.reducer;
