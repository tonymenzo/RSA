// main21.cc is a part of the PYTHIA event generator.
// Copyright (C) 2024 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Keywords: hadronization

// This is a simple test program.
// It illustrates how to feed in a single particle (including a resonance)
// or a toy parton-level configurations.

// Modified by: Tony Menzo
// Date: ------ 27-Nov-2024
// Summary:---- This program generates hadronization from a single string configuration amenable for  
// ------------ event reweighting. During event generation the program will output the hadron level 
// ------------ particle flow data, the accepted and rejected chains, the mT2 values and the fPrel 
// ------------ values. For more details refer to the README and/or arXiv:2411.02194.

#include "Pythia8/Pythia.h"
using namespace Pythia8;

//==========================================================================

// Single-particle gun. The particle must be a colour singlet.
// Input: flavour, energy, direction (theta, phi).
// If theta < 0 then random choice over solid angle.
// Optional final argument to put particle at rest => E = m.

void fillParticle(int id, double ee, double thetaIn, double phiIn,
  Event& event, ParticleData& pdt, Rndm& rndm, bool atRest = false,
  bool hasLifetime = false) {

  // Reset event record to allow for new event.
  event.reset();

  // Select particle mass; where relevant according to Breit-Wigner.
  double mm = pdt.mSel(id);
  double pp = sqrtpos(ee*ee - mm*mm);

  // Special case when particle is supposed to be at rest.
  if (atRest) {
    ee = mm;
    pp = 0.;
  }

  // Angles as input or uniform in solid angle.
  double cThe, sThe, phi;
  if (thetaIn >= 0.) {
    cThe = cos(thetaIn);
    sThe = sin(thetaIn);
    phi  = phiIn;
  } else {
    cThe = 2. * rndm.flat() - 1.;
    sThe = sqrtpos(1. - cThe * cThe);
    phi = 2. * M_PI * rndm.flat();
  }

  // Store the particle in the event record.
  int iNew = event.append( id, 1, 0, 0, pp * sThe * cos(phi),
    pp * sThe * sin(phi), pp * cThe, ee, mm);

  // Generate lifetime, to give decay away from primary vertex.
  if (hasLifetime) event[iNew].tau( event[iNew].tau0() * rndm.exp() );

}

//==========================================================================

// Simple method to do the filling of partons into the event record.

void fillPartons(int type, double ee, Event& event, ParticleData& pdt, Rndm& rndm) {

  // Reset event record to allow for new event.
  event.reset();

  // Information on a q qbar system, to be hadronized.
  if (type == 1 || type == 12) {
    int    id = 2;
    double mm = pdt.m0(id);
    double pp = sqrtpos(ee*ee - mm*mm);
    event.append(  id, 23, 101,   0, 0., 0.,  pp, ee, mm);
    event.append( -id, 23,   0, 101, 0., 0., -pp, ee, mm);

  // Information on a g g system, to be hadronized.
  } else if (type == 2 || type == 13) {
    event.append( 21, 23, 101, 102, 0., 0.,  ee, ee);
    event.append( 21, 23, 102, 101, 0., 0., -ee, ee);

  // Information on a g g g system, to be hadronized.
  } else if (type == 3) {
    event.append( 21, 23, 101, 102,        0., 0.,        ee, ee);
    event.append( 21, 23, 102, 103,  0.8 * ee, 0., -0.6 * ee, ee);
    event.append( 21, 23, 103, 101, -0.8 * ee, 0., -0.6 * ee, ee);

  // Information on a q q q junction system, to be hadronized.
  } else if (type == 4 || type == 5) {

    // Need a colour singlet mother parton to define junction origin.
    event.append( 1000022, -21, 0, 0, 2, 4, 0, 0,
                  0., 0., 1.01 * ee, 1.01 * ee);

    // The three endpoint q q q; the minimal system.
    double rt75 = sqrt(0.75);
    event.append( 2, 23, 1, 0, 0, 0, 101, 0,
                          0., 0., 1.01 * ee, 1.01 * ee);
    event.append( 2, 23, 1, 0, 0, 0, 102, 0,
                   rt75 * ee, 0., -0.5 * ee,        ee );
    event.append( 1, 23, 1, 0, 0, 0, 103, 0,
                  -rt75 * ee, 0., -0.5 * ee,        ee );

    // Define the qqq configuration as starting point for adding gluons.
    if (type == 5) {
      int colNow[4] = {0, 101, 102, 103};
      Vec4 pQ[4];
      pQ[1] = Vec4(0., 0., 1., 0.);
      pQ[2] = Vec4( rt75, 0., -0.5, 0.);
      pQ[3] = Vec4(-rt75, 0., -0.5, 0.);

      // Minimal cos(q-g opening angle), allows more or less nasty events.
      double cosThetaMin =0.;

      // Add a few gluons (almost) at random.
      for (int nglu = 0; nglu < 5; ++nglu) {
        int iq = 1 + int( 2.99999 * rndm.flat() );
        double px, py, pz, e, prod;
        do {
          e =  ee * rndm.flat();
          double cThe = 2. * rndm.flat() - 1.;
          double phi = 2. * M_PI * rndm.flat();
          px = e * sqrt(1. - cThe*cThe) * cos(phi);
          py = e * sqrt(1. - cThe*cThe) * sin(phi);
          pz = e * cThe;
          prod = ( px * pQ[iq].px() + py * pQ[iq].py() + pz * pQ[iq].pz() )
            / e;
        } while (prod < cosThetaMin);
        int colNew = 104 + nglu;
        event.append( 21, 23, 1, 0, 0, 0, colNew, colNow[iq],
          px, py, pz, e, 0.);
        colNow[iq] = colNew;
      }
      // Update daughter range of mother.
      event[1].daughters(2, event.size() - 1);

    }

  // Information on a q q qbar qbar dijunction system, to be hadronized.
  } else if (type >= 6) {

    // The two fictitious beam remnant particles; needed for junctions.
    event.append( 2212, -12, 0, 0, 3, 5, 0, 0, 0., 0., ee, ee, 0.);
    event.append(-2212, -12, 0, 0, 6, 8, 0, 0, 0., 0., ee, ee, 0.);

    // Opening angle between "diquark" legs.
    double theta = 0.2;
    double cThe = cos(theta);
    double sThe = sin(theta);

    // Set one colour depending on whether more gluons or not.
    int acol = (type == 6) ? 103 : 106;

    // The four endpoint q q qbar qbar; the minimal system.
    // Two additional fictitious partons to make up original beams.
    event.append(  2,   23, 1, 0, 0, 0, 101, 0,
                  ee * sThe, 0.,  ee * cThe, ee, 0.);
    event.append(  1,   23, 1, 0, 0, 0, 102, 0,
                 -ee * sThe, 0.,  ee * cThe, ee, 0.);
    event.append(  2, -21, 1, 0, 0, 0, 103, 0,
                         0., 0.,  ee       , ee, 0.);
    event.append( -2,   23, 2, 0, 0, 0, 0, 104,
                  ee * sThe, 0., -ee * cThe, ee, 0.);
    event.append( -1,   23, 2, 0, 0, 0, 0, 105,
                 -ee * sThe, 0., -ee * cThe, ee, 0.);
    event.append( -2, -21, 2, 0, 0, 0, 0, acol,
                         0., 0., -ee       , ee, 0.);

    // Add extra gluons on string between junctions.
    if (type == 7) {
      event.append( 21, 23, 8, 5, 0, 0, 103, 106, 0., ee, 0., ee, 0.);
    } else if (type == 8) {
      event.append( 21, 23, 8, 5, 0, 0, 103, 108, 0., ee, 0., ee, 0.);
      event.append( 21, 23, 8, 5, 0, 0, 108, 106, 0.,-ee, 0., ee, 0.);
    } else if (type == 9) {
      event.append( 21, 23, 8, 5, 0, 0, 103, 107, 0., ee, 0., ee, 0.);
      event.append( 21, 23, 8, 5, 0, 0, 107, 108, ee, 0., 0., ee, 0.);
      event.append( 21, 23, 8, 5, 0, 0, 108, 106, 0.,-ee, 0., ee, 0.);
    } else if (type == 10) {
      event.append( 21, 23, 8, 5, 0, 0, 103, 107, 0., ee, 0., ee, 0.);
      event.append( 21, 23, 8, 5, 0, 0, 107, 108, ee, 0., 0., ee, 0.);
      event.append( 21, 23, 8, 5, 0, 0, 108, 109, 0.,-ee, 0., ee, 0.);
      event.append( 21, 23, 8, 5, 0, 0, 109, 106,-ee, 0., 0., ee, 0.);
    }

  // No more cases: done.
  }
}

//==========================================================================

int main() {

  // Loop over kind of events to generate:
  // 0 = single-particle gun.
  // 1 = q qbar.
  // 2 = g g.
  // 3 = g g g.
  // 4 = minimal q q q junction topology.
  // 5 = q q q junction topology with gluons on the strings.
  // 6 = q q qbar qbar dijunction topology, no gluons.
  // 7 - 10 = ditto, but with 1 - 4 gluons on string between junctions.
  // 11 = single-resonance gun.
  // 12 = q qbar plus parton shower.
  // 13 = g g plus parton shower.
  // It is easy to edit the line below to only study one of them.
  int    type = 1;
  
  // Set particle species and energy for single-particle gun.
  int    idGun  = (type == 0) ? 15 : 25;
  double eeGun  = (type == 0) ? 20. : 125.;
  bool   atRest = (type == 0) ? false : true;

  // The single-particle gun produces a particle at the origin, and
  // by default decays it there. When hasLifetime = true instead a
  // finite lifetime is selected and used to generate a displaced
  // decay vertex.
  bool   hasLifetime = (type == 0) ? true : false;

  // Set typical energy per parton.
  double ee = 50.0;

  // Set number of events to generate and list.
  int nEvent = 10000;
  
  // Set a cutoff for the maximum number of rejections
  //int MAX_REJECT = 100;
  
  // Set data PATHs
  string hadron_PATH = "pgun_qqbar_finalTwo_hadrons_a_0.72_b_0.88_sigma_0.335_N_1e4.txt";
  string acceptReject_PATH = "pgun_qqbar_finalTwo_accept_reject_z_a_0.72_b_0.88_sigma_0.335_N_1e4.txt";
  string mT2_PATH = "pgun_qqbar_finalTwo_mT2_a_0.72_b_0.88_sigma_0.335_N_1e4.txt";
  string fPrel_PATH = "pgun_qqbar_finalTwo_fPrel_a_0.72_b_0.88_sigma_0.335_N_1e4.txt";

  // Generator; shorthand for event and particleData.
  Pythia pythia;
  Event& event           = pythia.event;
  ParticleData& pdt      = pythia.particleData;

  // Set the fragmentation parameters.
  //pythia.readString("StringZ:aLund = 0.68");   // Monash
  //pythia.readString("StringZ:bLund = 0.98");   // Monash
  //pythia.readString("StringPT:sigma = 0.335"); // Monash

  pythia.readString("StringZ:aLund = 0.72");   // Monash'
  pythia.readString("StringZ:bLund = 0.88");   // Monash'
  pythia.readString("StringPT:sigma = 0.335"); // Monash'

  // Auxiliary Pythia parameters
  pythia.readString("ProcessLevel:all = off");
  pythia.readString("HadronLevel:Decay = off");
  pythia.readString("StringFragmentation:TraceColours = on");

  // Set rng seed.
  pythia.readString("Random:setSeed = true");
  pythia.readString("Random:seed = 1");

  // Modify the flavor parameters such that only pions are allowed.
  pythia.readString("StringFlav:probQQtoQ = 0");
  pythia.readString("StringFlav:probStoUD = 0");
  pythia.readString("StringFlav:mesonUDvector = 0");
  pythia.readString("StringFlav:etaSup = 0");
  pythia.readString("StringFlav:etaPrimeSup = 0");
  pythia.readString("StringPT:enhancedFraction = 0");

  // Initialize.
  pythia.init();

  // Initialize event counter.
  int eventCounter = 0;

  // Begin of event loop.
  do {
    // Set up single particle, with random direction in solid angle.
    if (type == 0 || type == 11) fillParticle( idGun, eeGun, -1., 0.,
      event, pdt, pythia.rndm, atRest, hasLifetime);

    // Set up parton-level configuration.
    else fillPartons(type, ee, event, pdt, pythia.rndm);

    // Generate events. Quit if failure.
    if (!pythia.next()) {
      cout << "Event generation aborted prematurely, owing to error!\n";
      break;
    }
    
    // Print the total number of hadrons (not including the last two hadrons produced by finalTwo).
    int nHadrons = event.size() - 5;
    //cout << "The number of hadrons is: " << nHadrons << endl;

    // Read in accept and reject data - reject if multiplicity != number of accepted z
    ifstream rfile("acceptReject_i.txt", ios::in | ios::out);
    ifstream mT2file("mT2_i.txt", ios::in | ios::out);
    ifstream fPrelfile("fPrel_i.txt", ios::in | ios::out);
    string line;
    int nLines = 0;
    
    // Compute the number of lines in the file
    while (getline(rfile, line)) {  // Loop through each line in the file
      nLines++; // Incrementing line count for each line read
    }

    //cout << "The total number of lines in acceptReject_i.txt: " << nLines << endl;

    // Reset the read file
    rfile.clear();
    rfile.seekg(0);
    
    //cout << "The total number of lines in acceptReject_i.txt: " << nLines << endl;

    // Append the accepted and rejected values to the master file
    if (nLines == nHadrons) {
      // Create accept-reject file.
      ofstream ardatafile(acceptReject_PATH, ios::out | ios::in | ios::app);
      // Copy the event into the data file.
      while (getline(rfile, line)) {
        ardatafile << line << endl;
      }
      // Input a new line to separate events
      ardatafile << endl;
      ardatafile.close();

      // Create mT2 file.
      ofstream mT2datafile(mT2_PATH, ios::out | ios::in | ios::app);
      // Copy the event into the data file.
      while (getline(mT2file, line)) {
        mT2datafile << line << endl;
      }
      // Input a new line to separate events
      mT2datafile << endl;
      mT2datafile.close();

      // Output hadron level particle flow data.
      ofstream hadronfile(hadron_PATH, ios::out | ios::in | ios::app);
      for (int i = 3; i <= event.size() - 3; ++i) {
      // Convention (px, py, pz, E, m)
      hadronfile << event[i].px() << " " << event[i].py() << " " << event[i].pz() << " " << event[i].e() << " " << event[i].m() << endl;
      }
      // Input a new line to separate events
      hadronfile << endl;
      hadronfile.close();

      // Create fPrel file.
      ofstream fPreldatafile(fPrel_PATH, ios::out | ios::in | ios::app);
      // Copy the event into the data file.
      while (getline(fPrelfile, line)) {
        fPreldatafile << line << endl;
      }
      // Input a new line to separate events
      fPreldatafile << endl;
      fPreldatafile.close();

      // Iterate the event counter.
      eventCounter++;
    } else if (nLines != nHadrons) {
      // Write out accepted and rejected chains with some convention denoting the rejected chains.
      int lineCounter = 0;
      // Append to the file
      ofstream ardatafile(acceptReject_PATH, ios::out | ios::in | ios::app);
      // Append to file while the line counter is less than the difference between the number of lines and number of hadrons (iterate until we hit the accepted chain)
      while (lineCounter < nLines - (nHadrons)) {
        getline(rfile, line);
        ardatafile << line << endl;
        lineCounter++;
      }

      // Write out the accepted chain fragmentation, separating the accepted and rejected chains by the string '&'
      ardatafile << "&" << endl;
      // Append the rejected fragmentations
      while (getline(rfile, line)) {
        ardatafile << line << endl;
      }
        
      // Input a new line to separate events
      ardatafile << endl;
      ardatafile.close();

      // Write out mT2 for accepted and rejects chains with some convention denoting the rejected chains.
      lineCounter = 0;
      // Append to the file
      ofstream mT2datafile(mT2_PATH, ios::out | ios::in | ios::app);
      // Append to file while the line counter is less than the difference between the number of lines and number of hadrons (iterate until we hit the accepted chain)
      while (lineCounter < nLines - (nHadrons)) {
        getline(mT2file, line);
        mT2datafile << line << endl;
        lineCounter++;
      }

      // Write out the accepted chain fragmentation, separating the accepted and rejected chains by the string '&'
      mT2datafile << "&" << endl;
      // Append the rejected fragmentations
      while (getline(mT2file, line)) {
        mT2datafile << line << endl;
      }
        
      // Input a new line to separate events
      mT2datafile << endl;
      mT2datafile.close();

      // Output hadron level particle flow data.
      ofstream hadronfile(hadron_PATH, ios::out | ios::app);
      for (int i = 3; i <= event.size() - 3; ++i) {
      // Convention (px, py, pz, E, m)
      hadronfile << event[i].px() << " " << event[i].py() << " " << event[i].pz() << " " << event[i].e() << " " << event[i].m() << endl;
      }
      hadronfile << endl;
      hadronfile.close();

      // Write out mT2 for accepted and rejects chains with some convention denoting the rejected chains.
      lineCounter = 0;
      // Append to the file
      ofstream fPreldatafile(fPrel_PATH, ios::out | ios::in | ios::app);
      // Append to file while the line counter is less than the difference between the number of lines and number of hadrons (iterate until we hit the accepted chain)
      while (lineCounter < nLines - (nHadrons)) {
        getline(fPrelfile, line);
        fPreldatafile << line << endl;
        lineCounter++;
      }

      // Write out the accepted chain fragmentation, separating the accepted and rejected chains by the string '&'
      fPreldatafile << "&" << endl;
      // Append the rejected fragmentations
      while (getline(fPrelfile, line)) {
        fPreldatafile << line << endl;
      }
        
      // Input a new line to separate events
      fPreldatafile << endl;
      fPreldatafile.close();

      // Iterate the event counter.
      eventCounter++;
    }

    rfile.close(); // Close acceptReject_i.txt
    mT2file.close(); // Close mT2_i.txt
    fPrelfile.close(); // Close fPrel_i.txt

    // Clear the temporary accept-reject file
    ofstream cfragfile("acceptReject_i.txt", ios::out | ios::trunc);
    cfragfile.close();

    // Clear the temporary mT2 file
    ofstream cMT2file("mT2_i.txt", ios::out | ios::trunc);
    cMT2file.close();

    // Clear the temporary fPrel file
    ofstream cfPrelfile("fPrel_i.txt", ios::out | ios::trunc);
    cfPrelfile.close();

  // End of event loop.
  } while (eventCounter < nEvent);
}
