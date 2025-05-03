#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/opengym-module.h"
#include "ns3/mobility-module.h"
#include "ns3/internet-module.h"
#include "ns3/wifi-module.h"
#include "ns3/yans-wifi-helper.h"
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-module.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("WifiSimulation");

// Named constants for configuration, internal linkage
namespace
{
  constexpr uint32_t kNumSta = 5;          // number of stations
  constexpr double kSimulationTime = 10.0; // total simulation time (s)
  constexpr double kEnvStepTime = 1.0;     // gym step interval (s)
  constexpr uint16_t kOpenGymPort = 5555;  // OpenGym server port
  constexpr uint32_t kMaxQueueLen = 50;    // max queue length for observations
  constexpr uint32_t kMaxCw = 1023;        // max contention window for actions
}

// Variables for runtime statistics
static double totalThroughput = 0;
static uint32_t collisionCount = 0;
const std::vector<std::string> features = {"N_STAs", "Throughput", "Collision_probability"};

Ptr<OpenGymSpace> MyGetObservationSpace(void)
{
  uint32_t nodeNum = NodeList::GetNNodes();
  std::vector<uint32_t> shape = {nodeNum};
  Ptr<OpenGymMultiDiscreteSpace> space =
      CreateObject<OpenGymMultiDiscreteSpace>(shape, kMaxQueueLen);
  NS_LOG_UNCOND("MyGetObservationSpace: " << space);
  return space;
}

Ptr<WifiMacQueue> GetQueue(Ptr<Node> node)
{
  Ptr<NetDevice> dev = node->GetDevice(0);
  Ptr<WifiNetDevice> wifi_dev = DynamicCast<WifiNetDevice>(dev);
  Ptr<WifiMac> wifi_mac = wifi_dev->GetMac();
  PointerValue ptr;

  wifi_mac->GetAttribute("BE_Txop", ptr);
  Ptr<Txop> txop = ptr.Get<Txop>();
  Ptr<WifiMacQueue> queue = txop->GetWifiMacQueue();
  return queue;
}

Ptr<OpenGymSpace> MyGetActionSpace(void)
{
  uint32_t nodeNum = NodeList::GetNNodes();
  std::vector<uint32_t> shape = {nodeNum};
  Ptr<OpenGymMultiDiscreteSpace> space =
      CreateObject<OpenGymMultiDiscreteSpace>(shape, kMaxCw);
  NS_LOG_UNCOND("MyGetActionSpace: " << space);
  return space;
}

Ptr<OpenGymDataContainer> MyGetObservation(void)
{
  uint32_t nodeNum = NodeList::GetNNodes();
  std::vector<uint32_t> shape = {nodeNum};
  Ptr<OpenGymMultiDiscreteContainer> box =
      CreateObject<OpenGymMultiDiscreteContainer>(shape);

  for (auto it = NodeList::Begin(); it != NodeList::End(); ++it)
  {
    Ptr<Node> node = *it;
    Ptr<WifiMacQueue> queue = GetQueue(node);
    uint32_t value = queue->GetNPackets();
    box->AddValue(value);
  }

  NS_LOG_UNCOND("MyGetObservation: " << box);
  return box;
}

float MyGetReward(void)
{
  return (float)totalThroughput;
}

void PhyTxDrop(Ptr<const Packet> packet, double snr)
{
  collisionCount++;
}

void CalculateThroughput()
{
  std::cout << Simulator::Now().GetSeconds()
            << "s - Throughput: " << totalThroughput
            << " Mbps, Collisions: " << collisionCount
            << std::endl;
  totalThroughput = 0;
  Simulator::Schedule(Seconds(1.0), &CalculateThroughput);
}

bool SetCw(Ptr<Node> node, uint32_t cwMinValue = 0, uint32_t cwMaxValue = 0)
{
  Ptr<NetDevice> dev = node->GetDevice(0);
  Ptr<WifiNetDevice> wifi_dev = DynamicCast<WifiNetDevice>(dev);
  Ptr<WifiMac> wifi_mac = wifi_dev->GetMac();
  NS_ASSERT(wifi_mac != nullptr);
  PointerValue ptr;
  if (!wifi_mac->GetAttributeFailSafe("BE_Txop", ptr))
  {
    NS_LOG_UNCOND("Failed to get Txop");
    return false;
  }
  Ptr<Txop> txop = ptr.Get<Txop>();
  NS_ASSERT(txop != nullptr);

  // if both set to the same value then we have uniform backoff?
  if (cwMinValue != 0)
  {
    NS_LOG_DEBUG("Set CW min: " << cwMinValue);
    txop->SetMinCw(cwMinValue);
  }

  if (cwMaxValue != 0)
  {
    NS_LOG_DEBUG("Set CW max: " << cwMaxValue);
    txop->SetMaxCw(cwMaxValue);
  }
  return true;
}

bool MyExecuteActions(Ptr<OpenGymDataContainer> action)
{
  NS_LOG_UNCOND("MyExecuteActions: " << action);
  Ptr<OpenGymMultiDiscreteContainer> box =
      DynamicCast<OpenGymMultiDiscreteContainer>(action);
  std::vector<uint32_t> actionVector = box->GetData();

  uint32_t nodeNum = NodeList::GetNNodes();
  for (uint32_t i = 0; i < nodeNum; ++i)
  {
    Ptr<Node> node = NodeList::GetNode(i);
    uint32_t cwSize = actionVector.at(i);
    SetCw(node, cwSize, cwSize);
  }

  return true;
}

void ThroughputMonitor(std::string context,
                       Ptr<const Packet> packet,
                       double snr,
                       WifiMode mode,
                       WifiPreamble preamble)
{
  totalThroughput += (packet->GetSize() * 8.0) / 1e6; // bytes to Mbps
}

bool MyGetGameOver(void)
{
  bool isGameOver = false;
  NS_LOG_UNCOND("MyGetGameOver: " << isGameOver);
  return isGameOver;
}

void ScheduleNextStateRead(double envStepTime,
                           Ptr<OpenGymInterface> openGymInterface)
{
  Simulator::Schedule(Seconds(envStepTime), &ScheduleNextStateRead,
                      envStepTime, openGymInterface);
  openGymInterface->NotifyCurrentState();
}

int main(int argc, char *argv[])
{
  uint32_t nSta = kNumSta;
  double simulationTime = kSimulationTime;
  double envStepTime = kEnvStepTime;
  uint16_t openGymPort = kOpenGymPort;

  NodeContainer wifiStaNodes, wifiApNode;
  wifiStaNodes.Create(nSta);
  wifiApNode.Create(1);

  YansWifiChannelHelper channel = YansWifiChannelHelper::Default();
  YansWifiPhyHelper phy;
  phy.SetChannel(channel.Create());

  WifiHelper wifi;
  wifi.SetStandard(WIFI_STANDARD_80211n);

  WifiMacHelper mac;
  Ssid ssid("ns3-wifi");

  mac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(ssid),
              "ActiveProbing", BooleanValue(false), "QosSupported", BooleanValue(true));
  NetDeviceContainer staDevices = wifi.Install(phy, mac, wifiStaNodes);

  mac.SetType("ns3::ApWifiMac", "Ssid", SsidValue(ssid), "QosSupported", BooleanValue(true));
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

  Config::Connect("/NodeList/*/DeviceList/*/Phy/State/RxOk",
                  MakeCallback(&ThroughputMonitor));
  Config::Connect("/NodeList/*/DeviceList/*/Phy/State/TxDrop",
                  MakeCallback(&PhyTxDrop));

  Simulator::Schedule(Seconds(0.0), &CalculateThroughput);

  Ptr<OpenGymInterface> openGymInterface =
      CreateObject<OpenGymInterface>(openGymPort);
  openGymInterface->SetGetActionSpaceCb(
      MakeCallback(&MyGetActionSpace));
  openGymInterface->SetGetObservationSpaceCb(
      MakeCallback(&MyGetObservationSpace));
  openGymInterface->SetGetGameOverCb(
      MakeCallback(&MyGetGameOver));
  openGymInterface->SetGetObservationCb(
      MakeCallback(&MyGetObservation));
  openGymInterface->SetGetRewardCb(
      MakeCallback(&MyGetReward));
  openGymInterface->SetExecuteActionsCb(
      MakeCallback(&MyExecuteActions));

  Simulator::Schedule(Seconds(0.0), &ScheduleNextStateRead,
                      envStepTime, openGymInterface);
  Simulator::Stop(Seconds(simulationTime));
  Simulator::Run();
  Simulator::Destroy();

  return 0;
}
