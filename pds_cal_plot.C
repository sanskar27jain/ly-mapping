/*
* Alex Heindel
* Date: 7/22/26
* PDS Light-Yield Solution Plotter
* ---
*
* This ROOT macro reads a three-dimensional voxelized light-yield solution
* from a text file and produces summed projections, average projections, and
* individual two-dimensional slices of the reconstructed detector map.
*
* Input file format:
* The first value is the number of voxel divisions per axis, ndiv.
* The remaining ndiv^3 values are the reconstructed voxel light yields,
* Expressed in PE/MeV.
*
* IMPORTANT GEOMETRY REQUIREMENT:
* The coordinate bounds defined below by xmin/xmax, ymin/ymax, and
* zmin/zmax MUST exactly match the detector bounds used when the solution
* file was generated. The solution file stores only ndiv and the flattened
* voxel values; it does not store the physical coordinate bounds.
*
* Using different bounds here will assign the reconstructed values to
* incorrect physical positions and will produce geometrically incorrect
* projections and slices, even though the number of voxels may still match.
*
* Voxel ordering:
* The code assumes that the solution is flattened using the indexing
*
* ```
    index = ix + iy * ndiv + iz * ndiv^2
  ```
*
* where x varies fastest, followed by y, and then z. This ordering must also
* match the ordering used by the reconstruction code that wrote the solution.
*
* Output:
* * Summed YX, ZX, and YZ projections
* * Average YX, ZX, and YZ projections
* * One two-dimensional slice for each voxel layer along each axis
*
* Before running:
* 1. Set the input solution-file path.
* 2. Set xmin/xmax, ymin/ymax, and zmin/zmax to the reconstruction bounds.
* 3. Ensure that the output directory exists.
*
* This macro is intended as an example and may need to be adapted for other
* detector geometries, solution formats, voxel orderings, or plotting needs.
*/


#include <iostream>
#include <cmath>
#include <array>
#include <limits>
#include <vector>

#include "TFile.h"
#include "TTree.h"
#include "TH2D.h"
#include "TGraph.h"
#include "TCanvas.h"

#include <fstream>

void pds_cal_plot()
{
	float xmin = -400;
	float xmax = 400;
	float ymin = -425;
	float ymax = 425;
	float zmin = -300;
	float zmax = 600;

	std::vector<double> x;
	std::ifstream x_in("angelo2_12x12x12_60000evts_lambda3e6.txt");

	if (!x_in) {
    	std::cerr << "Error: could not open input file\n";
    	return;
	}

	int ndiv = 0;
	x_in >> ndiv;
	int arr_size = pow(ndiv, 3);
	
	double val;
	while (x_in >> val) {
	    x.push_back(val);
	}
	

	// Result is in photons/MeV	
	//std::cout << "Solution x:\n" << x << std::endl;
	
	gStyle->SetOptStat(0);

	TH3D* h_xyz = new TH3D("h_xyz",
                       "3D voxel map;X (cm);Y (cm);Z (cm)",
                       ndiv, xmin, xmax,
                       ndiv, ymin, ymax,
                       ndiv, zmin, zmax);

	for (int iz = 0; iz < ndiv; ++iz) 
	{
        for (int iy = 0; iy < ndiv; ++iy)
		{
            for (int ix = 0; ix < ndiv; ++ix)
			{
				//if (x[ix + iy*ndiv + iz*pow(ndiv,2)] < 3000){
					h_xyz->SetBinContent(iy + 1, ix + 1, iz + 1, x[ix + iy*ndiv + iz*pow(ndiv,2)]);  // Bin numbers start from 1 in ROOT
				//}
        	}	
		}
    }


    // Summed + average projections
    auto patchZeros = [](TH2D* h)
    {
        for (int ix = 1; ix <= h->GetNbinsX(); ++ix)
        {
            for (int iy = 1; iy <= h->GetNbinsY(); ++iy)
            {
                if (h->GetBinContent(ix, iy) == 0.0)
                {
                    h->SetBinContent(ix, iy, 0.0001);
                }
            }
        }
    };

    auto makeCanvas = [](const char* name)
    {
        TCanvas* c = new TCanvas(name, name, 800, 600);
        c->SetRightMargin(0.15);
        c->SetLeftMargin(0.12);
        c->SetBottomMargin(0.12);
        c->SetTopMargin(0.08);
        return c;
    };

    // -------------------------
    // YX summed projection
    // -------------------------
    TH2D* h_yx = (TH2D*)h_xyz->Project3D("xy");
    h_yx->SetName("h_yx");
    h_yx->SetTitle("YX summed projection;Y (cm);X (cm)");
    h_yx->SetStats(0);
    patchZeros(h_yx);

    TCanvas* c_yx = makeCanvas("c_yx");
    h_yx->SetMinimum(0.0);
    h_yx->GetZaxis()->SetTitle("PE/MeV");
    h_yx->GetZaxis()->CenterTitle();
    h_yx->GetZaxis()->SetTitleOffset(1.25);
    h_yx->GetZaxis()->RotateTitle(true);
    h_yx->Draw("COLZ");
    c_yx->SaveAs(Form("test_plot/sum_h_yx_%dx%d_nnls_test.png", ndiv, ndiv));
    c_yx->SaveAs(Form("test_plot/sum_h_yx_%dx%d_nnls_test.pdf", ndiv, ndiv));

    // YX average projection
    TH2D* h_yx_avg = (TH2D*)h_yx->Clone("h_yx_avg");
    h_yx_avg->Scale(1.0 / ndiv);
    h_yx_avg->SetTitle(";Y (cm);X (cm)");
    h_yx_avg->SetStats(0);
    patchZeros(h_yx_avg);

    TCanvas* c_yx_avg = makeCanvas("c_yx_avg");
    h_yx_avg->SetMinimum(0.0);
    h_yx_avg->GetZaxis()->SetTitle("PE/MeV");
    h_yx_avg->GetZaxis()->CenterTitle();
    h_yx_avg->GetZaxis()->SetTitleOffset(1.25);
    h_yx_avg->GetZaxis()->RotateTitle(true);
    h_yx_avg->Draw("COLZ");
    c_yx_avg->SaveAs(Form("test_plot/avg_h_yx_%dx%d_nnls_test.png", ndiv, ndiv));
    c_yx_avg->SaveAs(Form("test_plot/avg_h_yx_%dx%d_nnls_test.pdf", ndiv, ndiv));

    // -------------------------
    // ZX summed projection
    // -------------------------
    TH2D* h_zx = (TH2D*)h_xyz->Project3D("xz");
    h_zx->SetName("h_zx");
    h_zx->SetTitle("ZX summed projection;Z (cm);X (cm)");
    h_zx->SetStats(0);
    patchZeros(h_zx);

    TCanvas* c_zx = makeCanvas("c_zx");
    h_zx->SetMinimum(0.0);
    h_zx->GetZaxis()->SetTitle("PE/MeV");
    h_zx->GetZaxis()->CenterTitle();
    h_zx->GetZaxis()->SetTitleOffset(1.25);
    h_zx->GetZaxis()->RotateTitle(true);
    h_zx->Draw("COLZ");
    c_zx->SaveAs(Form("test_plot/sum_h_zx_%dx%d_nnls_test.png", ndiv, ndiv));
    c_zx->SaveAs(Form("test_plot/sum_h_zx_%dx%d_nnls_test.pdf", ndiv, ndiv));

    // ZX average projection
    TH2D* h_zx_avg = (TH2D*)h_zx->Clone("h_zx_avg");
    h_zx_avg->Scale(1.0 / ndiv);
    h_zx_avg->SetTitle(";Z (cm);X (cm)");
    h_zx_avg->SetStats(0);
    patchZeros(h_zx_avg);

    TCanvas* c_zx_avg = makeCanvas("c_zx_avg");
    h_zx_avg->SetMinimum(0.0);
    h_zx_avg->GetZaxis()->SetTitle("PE/MeV");
    h_zx_avg->GetZaxis()->CenterTitle();
    h_zx_avg->GetZaxis()->SetTitleOffset(1.25);
    h_zx_avg->GetZaxis()->RotateTitle(true);
    h_zx_avg->Draw("COLZ");
    c_zx_avg->SaveAs(Form("test_plot/avg_h_zx_%dx%d_nnls_test.png", ndiv, ndiv));
    c_zx_avg->SaveAs(Form("test_plot/avg_h_zx_%dx%d_nnls_test.pdf", ndiv, ndiv));

    // -------------------------
    // YZ summed projection
    // -------------------------
    TH2D* h_yz = (TH2D*)h_xyz->Project3D("zy");
    h_yz->SetName("h_yz");
    h_yz->SetTitle("YZ summed projection;Y (cm);Z (cm)");
    h_yz->SetStats(0);
    patchZeros(h_yz);

    TCanvas* c_yz = makeCanvas("c_yz");
    h_yz->SetMinimum(0.0);
    h_yz->GetZaxis()->SetTitle("PE/MeV");
    h_yz->GetZaxis()->CenterTitle();
    h_yz->GetZaxis()->SetTitleOffset(1.25);
    h_yz->GetZaxis()->RotateTitle(true);
    h_yz->Draw("COLZ");
    c_yz->SaveAs(Form("test_plot/sum_h_yz_%dx%d_nnls_test.png", ndiv, ndiv));
    c_yz->SaveAs(Form("test_plot/sum_h_yz_%dx%d_nnls_test.pdf", ndiv, ndiv));

    // YZ average projection
    TH2D* h_yz_avg = (TH2D*)h_yz->Clone("h_yz_avg");
    h_yz_avg->Scale(1.0 / ndiv);
    h_yz_avg->SetTitle(";Y (cm);Z (cm)");
    h_yz_avg->SetStats(0);
    patchZeros(h_yz_avg);

    TCanvas* c_yz_avg = makeCanvas("c_yz_avg");
    h_yz_avg->SetMinimum(0.0);
    h_yz_avg->GetZaxis()->SetTitle("PE/MeV");
    h_yz_avg->GetZaxis()->CenterTitle();
    h_yz_avg->GetZaxis()->SetTitleOffset(1.25);
    h_yz_avg->GetZaxis()->RotateTitle(true);
    h_yz_avg->Draw("COLZ");
    c_yz_avg->SaveAs(Form("test_plot/avg_h_yz_%dx%d_nnls_test.png", ndiv, ndiv));
    c_yz_avg->SaveAs(Form("test_plot/avg_h_yz_%dx%d_nnls_test.pdf", ndiv, ndiv));

	// Slices
	TCanvas* c_yx_slice = new TCanvas("c_yx_slice", "Canvas", 800, 600);
	c_yx_slice->SetRightMargin(0.15);
	c_yx_slice->SetLeftMargin(0.12);
	c_yx_slice->SetBottomMargin(0.12);
	c_yx_slice->SetTopMargin(0.08);

	for (int iz = 1; iz <= ndiv; ++iz)
	{
	    h_xyz->GetZaxis()->SetRange(iz, iz);

	    TH2D* h_yx_slice = (TH2D*)h_xyz->Project3D("xy");

	    h_yx_slice->SetTitle(Form("Y vs X Slice %d;Y (cm);X (cm)", iz - 1));
	    h_yx_slice->SetStats(0);

	    for (int bx = 1; bx <= h_yx_slice->GetNbinsX(); ++bx) {
	        for (int by = 1; by <= h_yx_slice->GetNbinsY(); ++by) {
	            if (h_yx_slice->GetBinContent(bx, by) == 0.0) {
	                h_yx_slice->SetBinContent(bx, by, 0.0001);
	            }
	        }
	    }

	    c_yx_slice->cd();
	    h_yx_slice->SetMinimum(0.0);
	    h_yx_slice->GetZaxis()->SetTitle("PE/MeV");
	    h_yx_slice->GetZaxis()->CenterTitle();
	    h_yx_slice->GetZaxis()->SetTitleOffset(1.25);
	    h_yx_slice->GetZaxis()->RotateTitle(true);
	    h_yx_slice->Draw("COLZ");

	    c_yx_slice->SaveAs(Form("test_plot/h_yx_%dx%d_nnls_slice_%d.png", ndiv, ndiv, iz - 1));

	    delete h_yx_slice;
	}

	// reset axis range afterward
	h_xyz->GetZaxis()->SetRange(0, 0);

	TCanvas* c_zx_slice = new TCanvas("c_zx_slice", "Canvas", 800, 600);
	c_zx_slice->SetRightMargin(0.15);
	c_zx_slice->SetLeftMargin(0.12);
	c_zx_slice->SetBottomMargin(0.12);
	c_zx_slice->SetTopMargin(0.08);

	for (int iy = 1; iy <= ndiv; ++iy)
	{
	    h_xyz->GetYaxis()->SetRange(iy, iy);

	    TH2D* h_zx_slice = (TH2D*)h_xyz->Project3D("xz");

	    h_zx_slice->SetTitle(Form("Z vs X Slice %d;Z (cm);X (cm)", iy - 1));
	    h_zx_slice->SetStats(0);

	    for (int bx = 1; bx <= h_zx_slice->GetNbinsX(); ++bx) {
	        for (int by = 1; by <= h_zx_slice->GetNbinsY(); ++by) {
	            if (h_zx_slice->GetBinContent(bx, by) == 0.0) {
	                h_zx_slice->SetBinContent(bx, by, 0.0001);
	            }
	        }
	    }

	    c_zx_slice->cd();
	    h_zx_slice->SetMinimum(0.0);
	    h_zx_slice->GetZaxis()->SetTitle("PE/MeV");
	    h_zx_slice->GetZaxis()->CenterTitle();
	    h_zx_slice->GetZaxis()->SetTitleOffset(1.25);
	    h_zx_slice->GetZaxis()->RotateTitle(true);
	    h_zx_slice->Draw("COLZ");

	    c_zx_slice->SaveAs(Form("test_plot/h_zx_%dx%d_nnls_slice_%d.png", ndiv, ndiv, iy - 1));

	    delete h_zx_slice;
	}

	// reset axis range afterward
	h_xyz->GetYaxis()->SetRange(0, 0);

	TCanvas* c_yz_slice = new TCanvas("c_yz_slice", "Canvas", 800, 600);
	c_yz_slice->SetRightMargin(0.15);
	c_yz_slice->SetLeftMargin(0.12);
	c_yz_slice->SetBottomMargin(0.12);
	c_yz_slice->SetTopMargin(0.08);

	for (int ix = 1; ix <= ndiv; ++ix)
	{
	    h_xyz->GetXaxis()->SetRange(ix, ix);

	    TH2D* h_yz_slice = (TH2D*)h_xyz->Project3D("zy");

	    h_yz_slice->SetTitle(Form("Y vs Z Slice %d;Y (cm);Z (cm)", ix - 1));
	    h_yz_slice->SetStats(0);

	    for (int bx = 1; bx <= h_yz_slice->GetNbinsX(); ++bx) {
	        for (int by = 1; by <= h_yz_slice->GetNbinsY(); ++by) {
	            if (h_yz_slice->GetBinContent(bx, by) == 0.0) {
	                h_yz_slice->SetBinContent(bx, by, 0.0001);
	            }
	        }
	    }

	    c_yz_slice->cd();
	    h_yz_slice->SetMinimum(0.0);
	    h_yz_slice->GetZaxis()->SetTitle("PE/MeV");
	    h_yz_slice->GetZaxis()->CenterTitle();
	    h_yz_slice->GetZaxis()->SetTitleOffset(1.25);
	    h_yz_slice->GetZaxis()->RotateTitle(true);
	    h_yz_slice->Draw("COLZ");

	    c_yz_slice->SaveAs(Form("test_plot/h_yz_%dx%d_nnls_slice_%d.png", ndiv, ndiv, ix - 1));

	    delete h_yz_slice;
	}

	// reset axis range afterward
	h_xyz->GetXaxis()->SetRange(0, 0);

	x_in.close();    
}
