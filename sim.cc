#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/internet-module.h"
#include "ns3/wifi-module.h"
#include "ns3/yans-wifi-helper.h"
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-module.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("WifiSimulation");

// Variables for statistics
double totalThroughput = 0;
uint32_t collisionCount = 0;

void PhyTxDrop(Ptr<const Packet> packet, double snr)
{
    collisionCount++;  // Increase collision count on dropped packets
}

void CalculateThroughput()
{
    std::cout << Simulator::Now().GetSeconds() << "s - Throughput: " << totalThroughput << " Mbps, Collisions: " << collisionCount << std::endl;
    totalThroughput = 0; // Reset for next interval
    Simulator::Schedule(Seconds(1.0), &CalculateThroughput);
}

void ThroughputMonitor(std::string context, Ptr<const Packet> packet, double snr, WifiMode mode, WifiPreamble preamble)
{
    totalThroughput += (packet->GetSize() * 8.0) / (1e6); // Convert bytes to Mbps
}

int main(int argc, char *argv[])
{
    uint32_t nSta = 5;
    double simulationTime = 10.0; // in seconds

    NodeContainer wifiStaNodes, wifiApNode;
    wifiStaNodes.Create(nSta);
    wifiApNode.Create(1);

    YansWifiChannelHelper channel = YansWifiChannelHelper::Default();
    YansWifiPhyHelper phy;
    phy.SetChannel(channel.Create());

    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211n);
    
    WifiMacHelper mac;
    Ssid ssid = Ssid("ns3-wifi");

    mac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(ssid), "ActiveProbing", BooleanValue(false));
    NetDeviceContainer staDevices = wifi.Install(phy, mac, wifiStaNodes);

    mac.SetType("ns3::ApWifiMac", "Ssid", SsidValue(ssid));
    NetDeviceContainer apDevice = wifi.Install(phy, mac, wifiApNode);

    MobilityHelper mobility;
    mobility.SetPositionAllocator("ns3::GridPositionAllocator",
                                  "MinX", DoubleValue(0.0),
                                  "MinY", DoubleValue(0.0),
                                  "DeltaX", DoubleValue(5.0),
                                  "DeltaY", DoubleValue(10.0),
                                  "GridWidth", UintegerValue(3),
                                  "LayoutType", StringValue("RowFirst"));
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(wifiStaNodes);
    mobility.Install(wifiApNode);

    InternetStackHelper stack;
    stack.Install(wifiStaNodes);
    stack.Install(wifiApNode);

    Ipv4AddressHelper address;
    address.SetBase("192.168.1.0", "255.255.255.0");
    Ipv4InterfaceContainer staInterfaces = address.Assign(staDevices);
    Ipv4InterfaceContainer apInterface = address.Assign(apDevice);

    UdpServerHelper server(9);
    ApplicationContainer serverApp = server.Install(wifiApNode.Get(0));
    serverApp.Start(Seconds(1.0));
    serverApp.Stop(Seconds(simulationTime));

    UdpClientHelper client(apInterface.GetAddress(0), 9);
    client.SetAttribute("MaxPackets", UintegerValue(100000));
    client.SetAttribute("Interval", TimeValue(MilliSeconds(1)));
    client.SetAttribute("PacketSize", UintegerValue(1024));

    ApplicationContainer clientApps;
    for (uint32_t i = 0; i < nSta; ++i)
    {
        clientApps.Add(client.Install(wifiStaNodes.Get(i)));
    }
    clientApps.Start(Seconds(2.0));
    clientApps.Stop(Seconds(simulationTime));

    Config::Connect("/NodeList/*/DeviceList/*/Phy/State/RxOk", MakeCallback(&ThroughputMonitor));
    
    Simulator::Schedule(Seconds(1.0), &CalculateThroughput);

    Simulator::Stop(Seconds(simulationTime));
    Simulator::Run();
    Simulator::Destroy();

    return 0;
}

